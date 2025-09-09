# optimizer/objective.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from OCCS.optimizer.utils import (
    load_two_row_csv,
    resample_to_ref,
    normalize_shape,
    huber_loss,
    make_band_weights,
)

@dataclass
class ObjectiveConfig:
    """
    标量目标函数的可调参数。
    默认先走“最小可行”配置：小范围对齐 + L2/Huber 距离 + （可选）分区加权。
    """
    # 对齐范围与步长（步长默认等于参考网格间距）
    delta_max_nm: float = 0.10
    delta_step_nm: Optional[float] = None

    # 稳健损失
    use_huber: bool = True
    huber_kappa: float = 0.8  # 归一尺度下的转折点

    # 分区加权（若不提供 weights，则使用均匀权重）
    weights: Optional[np.ndarray] = None

    # 是否在对齐后拟合幅度/偏置（alpha, beta）以进一步对齐整体尺度
    fit_gain_bias: bool = False

class CurveObjective:
    """
    将“整条响应曲线是否接近理想曲线”压缩为一个数值 y（越小越好）。
    使用：
        obj = create_objective_from_csv('data/ideal_waveform.csv', M=1001, ...)
        y, diag = obj(lambda_raw, s_raw)
    """
    def __init__(self,
                 lambda_ref: np.ndarray,
                 target_ref: np.ndarray,
                 config: Optional[ObjectiveConfig] = None):
        self.lambda_ref = np.asarray(lambda_ref, dtype=np.float64).ravel()
        self.target_ref = np.asarray(target_ref, dtype=np.float64).ravel()
        if self.lambda_ref.size != self.target_ref.size:
            raise ValueError("lambda_ref and target_ref length mismatch.")
        self.config = config or ObjectiveConfig()

        # 预先归一化理想曲线到形状尺度（对比时使用相同处理）
        self._target_norm = normalize_shape(self.target_ref)

        # 参考网格间距用于默认对齐步长（转换到 nm 单位存储）
        if self.config.delta_step_nm is None:
            if self.lambda_ref.size < 2:
                raise ValueError("lambda_ref needs at least 2 points to derive step.")
            # lambda_ref 与模拟数据通常以米为单位，需要转换为 nm
            self.config.delta_step_nm = float(self.lambda_ref[1] - self.lambda_ref[0]) * 1e9

        # 权重检查
        if self.config.weights is not None:
            w = np.asarray(self.config.weights, dtype=np.float64).ravel()
            if w.size != self.lambda_ref.size:
                raise ValueError("weights length must match lambda_ref length.")
            self._weights = w / (w.sum() + 1e-12)
        else:
            self._weights = None

    def __call__(self,
                 lambda_raw: np.ndarray,
                 s_raw: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        计算标量目标值 y 以及诊断信息（用于日志与可视化）。
        """
        cfg = self.config

        # 1) 重采样到参考网格 + 形状归一
        s_ref = resample_to_ref(lambda_raw, s_raw, self.lambda_ref)
        s_norm = normalize_shape(s_ref)

        # 2) 小范围波长对齐（离散平移）
        # 将 nm 配置转换到与 lambda_ref 相同的单位（米）
        step = cfg.delta_step_nm * 1e-9
        delta_max = cfg.delta_max_nm * 1e-9
        if step <= 0:
            raise ValueError("delta_step_nm must be positive")
        if step > delta_max:
            deltas = np.array([0.0])
        else:
            deltas = np.arange(-delta_max, delta_max + 1e-12, step)
        best_loss = None
        best_delta = 0.0
        best_s_aligned = s_norm

        for dlt in deltas:
            # 左右边界外推：用端点值延拓（简单稳健）
            s_shift = np.interp(self.lambda_ref,
                                self.lambda_ref + dlt,
                                s_norm,
                                left=s_norm[0], right=s_norm[-1])

            # 2.1 可选：对齐后再拟合 alpha,beta 以衔接幅度/偏置（多数情况下可关闭）
            if cfg.fit_gain_bias:
                A = np.vstack([s_shift, np.ones_like(s_shift)]).T  # [s, 1]
                # 最小二乘 [alpha, beta]
                try:
                    x, *_ = np.linalg.lstsq(A, self._target_norm, rcond=None)
                    alpha, beta = x[0], x[1]
                    s_cmp = alpha * s_shift + beta
                except np.linalg.LinAlgError:
                    s_cmp = s_shift
            else:
                s_cmp = s_shift

            # 3) 稳健距离（Huber 或 L2）+ 可选权重
            e = s_cmp - self._target_norm
            if cfg.use_huber:
                pointwise = huber_loss(e, cfg.huber_kappa)
            else:
                pointwise = e ** 2

            if self._weights is None:
                loss = float(pointwise.mean())
            else:
                loss = float((self._weights * pointwise).sum())

            if (best_loss is None) or (loss < best_loss):
                best_loss = loss
                best_delta = float(dlt)
                best_s_aligned = s_cmp

        diag = {
            # 将最优平移量以 nm 汇报
            "delta_nm": best_delta * 1e9,
            "lambda_ref": self.lambda_ref,
            "s_ref": s_ref,                # 重采样但未对齐
            "s_aligned": best_s_aligned,   # 对齐(+可选alpha,beta)后的谱（已归一化尺度）
            "target_norm": self._target_norm
        }
        return float(best_loss), diag

def create_objective_from_csv(target_csv_path: str | Path,
                              M: int = 1001,
                              lambda_min: Optional[float] = None,
                              lambda_max: Optional[float] = None,
                              passband: Optional[Tuple[float, float]] = None,
                              transition: Optional[Tuple[float, float]] = None,
                              stopband: Optional[Tuple[float, float]] = None,
                              config: Optional[ObjectiveConfig] = None) -> CurveObjective:
    """
    读取两行CSV的理想曲线，构造统一参考网格与（可选）分区权重，返回可调用的目标函数对象。
    - M：参考网格点数
    - lambda_min/max：不提供则取CSV的范围
    - passband/transition/stopband：若提供，将自动生成分区权重（通带>过渡带>阻带）
    """
    lam_t, t_raw = load_two_row_csv(target_csv_path)

    # 生成参考网格
    lam_lo = float(lam_t.min() if lambda_min is None else lambda_min)
    lam_hi = float(lam_t.max() if lambda_max is None else lambda_max)
    lambda_ref = np.linspace(lam_lo, lam_hi, int(M))

    # 将理想曲线重采样到参考网格
    t_ref = resample_to_ref(lam_t, t_raw, lambda_ref)

    # 配置权重
    cfg = config or ObjectiveConfig()
    if cfg.weights is None and any(v is not None for v in (passband, transition, stopband)):
        cfg = dataclass_replace(cfg, weights=make_band_weights(lambda_ref, passband, transition, stopband))

    return CurveObjective(lambda_ref=lambda_ref, target_ref=t_ref, config=cfg)


class HardwareObjective:
    """Wrap :class:`CurveObjective` with a hardware interface.

    The returned callable takes a voltage vector, applies it to ``hardware`` and
    evaluates the optical response against the reference waveform.
    """

    def __init__(self, hardware, curve_obj: CurveObjective) -> None:
        self.hardware = hardware
        self.curve_obj = curve_obj
        self.wavelength = np.asarray(getattr(hardware, "wavelength"), dtype=float)

    def __call__(self, volts: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        volts = np.asarray(volts, dtype=float)
        self.hardware.apply_voltage(volts)
        signal = self.hardware.get_simulated_response()
        loss, diag = self.curve_obj(self.wavelength, signal)
        diag["volts"] = volts
        return loss, diag


def create_hardware_objective(
    hardware,
    target_csv_path: str | Path,
    M: int = 1001,
    lambda_min: Optional[float] = None,
    lambda_max: Optional[float] = None,
    passband: Optional[Tuple[float, float]] = None,
    transition: Optional[Tuple[float, float]] = None,
    stopband: Optional[Tuple[float, float]] = None,
    config: Optional[ObjectiveConfig] = None,
) -> HardwareObjective:
    """Convenience helper to build a hardware-coupled objective.

    Parameters mirror :func:`create_objective_from_csv` with an additional
    ``hardware`` argument providing ``apply_voltage`` and
    ``get_simulated_response`` methods as implemented by
    :class:`OCCS.connector.MockHardware`.
    """

    curve_obj = create_objective_from_csv(
        target_csv_path,
        M=M,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        passband=passband,
        transition=transition,
        stopband=stopband,
        config=config,
    )
    return HardwareObjective(hardware, curve_obj)

# 简易的 dataclass 替换（Python 3.10 无 dataclasses.replace 引入）
def dataclass_replace(cfg: ObjectiveConfig, **kwargs) -> ObjectiveConfig:
    data = cfg.__dict__.copy()
    data.update(kwargs)
    return ObjectiveConfig(**data)

# ---------- 下面是一个最小用例（可留作注释/自测） ----------
if __name__ == "__main__":
    try:
        from OCCS.connector.mock_hardware import MockHardware

        # 构造一个简易硬件和目标函数（使用自带的 ideal_waveform.csv）
        lam = np.linspace(1.55e-6, 1.56e-6, 200)
        hw = MockHardware(dac_size=3, wavelength=lam)
        obj = create_hardware_objective(
            hw,
            target_csv_path=Path(__file__).parent.parent / "data" / "ideal_waveform.csv",
            M=200,
        )

        # 给定电压后评估目标函数
        volts = np.zeros(hw.dac_size)
        y, diag = obj(volts)
        print(f"Objective y = {y:.6f}, best delta = {diag['delta_nm']:.4f} nm")
    except Exception as e:  # pragma: no cover - 仅用于开发自测
        print("Self-test skipped or failed:", e)
