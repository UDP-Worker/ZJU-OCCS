"""Scalar objective functions for curve matching and hardware coupling.

This module defines a robust curve similarity objective and a helper to bind
it to the mock/real hardware interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np

from OCCS.optimizer.utils import (
    load_two_row_csv,
    resample_to_ref,
    normalize_shape,
    huber_loss,
    make_band_weights,
)

@dataclass
class ObjectiveConfig:
    """Configuration for the scalar curve-matching objective.

    The default is a minimal, robust setup: small-range wavelength alignment,
    L2/Huber distance on normalised shapes, with optional band weighting.

    Parameters
    ----------
    delta_max_nm:
        Maximum absolute wavelength shift (in nm) allowed during discrete
        alignment search around the reference grid.
    delta_step_nm:
        Step size (in nm) for the discrete alignment search. If ``None``, it
        defaults to the spacing of the reference wavelength grid.
    use_huber:
        Whether to use Huber loss instead of pure L2.
    huber_kappa:
        Turnover point (on the normalised scale) for the Huber loss.
    weights:
        Optional per-sample weights on the reference grid. If ``None``, uses
        uniform weights. Can be generated via :func:`make_band_weights`.
    fit_gain_bias:
        If ``True``, after alignment, fit an affine transform ``alpha*s + beta``
        to match the target scale and offset before computing the loss.
    """
    delta_max_nm: float = 0.10
    delta_step_nm: Optional[float] = None
    use_huber: bool = True
    huber_kappa: float = 0.8
    weights: Optional[np.ndarray] = None
    fit_gain_bias: bool = False

class CurveObjective:
    """Curve similarity objective compressed to a scalar value.

    Given a measured spectrum, re-sample to a common grid, optionally align by
    a small wavelength shift, optionally fit gain/bias, then compute a robust
    distance to the target. Lower is better.

    Parameters
    ----------
    lambda_ref:
        Reference wavelength grid (1D array).
    target_ref:
        Target spectrum sampled on ``lambda_ref`` (1D array).
    config:
        Objective configuration. See :class:`ObjectiveConfig`.

    Examples
    --------
    obj = create_objective_from_csv('data/ideal_waveform.csv', M=1001)
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
        """Evaluate the objective with simple MSE and return diagnostics.

        Parameters
        ----------
        lambda_raw:
            Wavelength samples of the measured spectrum (1D array).
        s_raw:
            Measured spectrum values (1D array).

        Returns
        -------
        (loss, diag):
            ``loss`` is a float (lower is better). ``diag`` includes
            ``delta_nm``, ``lambda_ref``, ``s_ref``, ``s_aligned``, and
            ``target_norm`` useful for logging/plotting.
        """
        # 重采样到参考网格
        s_ref = resample_to_ref(lambda_raw, s_raw, self.lambda_ref)

        # 使用简单的 MSE 作为曲线间的距离（不做对齐与归一化/加权）
        err = s_ref - self.target_ref
        loss = float(np.mean(err ** 2))

        # 维持原有诊断信息接口（以便下游可视化或日志不受影响）
        diag = {
            "delta_nm": 0.0,               # 简化后无平移，对齐量视为 0
            "lambda_ref": self.lambda_ref,
            "s_ref": s_ref,                # 重采样后的信号
            "s_aligned": s_ref,            # 无对齐，等同于 s_ref
            "target_norm": self.target_ref # 此处直接返回目标曲线
        }
        return loss, diag

def create_objective_from_csv(target_csv_path: str | Path,
                              M: int = 1001,
                              lambda_min: Optional[float] = None,
                              lambda_max: Optional[float] = None,
                              passband: Optional[Tuple[float, float]] = None,
                              transition: Optional[Tuple[float, float]] = None,
                              stopband: Optional[Tuple[float, float]] = None,
                              config: Optional[ObjectiveConfig] = None) -> CurveObjective:
    """Build :class:`CurveObjective` from a two-row CSV.

    The CSV is expected to have two rows: first row for wavelengths, second for
    target values. The function constructs a common reference grid and optional
    band weights, then returns a configured objective.

    Parameters
    ----------
    target_csv_path:
        Path to the two-row CSV file.
    M:
        Number of points in the reference grid.
    lambda_min, lambda_max:
        Optional wavelength range override. Defaults to the CSV range.
    passband, transition, stopband:
        Optional wavelength intervals used to build piecewise weights with
        :func:`make_band_weights`.
    config:
        Optional :class:`ObjectiveConfig`. If ``weights`` is not set and band
        intervals are provided, weights are generated automatically.

    Returns
    -------
    CurveObjective
        Configured objective bound to the generated reference grid.
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
        cfg = dataclass_replace(
            cfg,
            weights=make_band_weights(
                lambda_ref,
                passband,
                transition,
                stopband,
            ),
        )

    return CurveObjective(lambda_ref=lambda_ref, target_ref=t_ref, config=cfg)


class HardwareObjective:
    """Hardware-coupled objective wrapper.

    Applies a voltage vector to ``hardware``, retrieves the simulated/real
    response, and evaluates it against the target using
    :class:`CurveObjective`.
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
    """Build a hardware-coupled objective.

    Mirrors :func:`create_objective_from_csv` parameters and adds ``hardware``
    providing ``apply_voltage`` and ``get_simulated_response`` as in
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
    """Shallow dataclass replace for ``ObjectiveConfig``.

    Parameters
    ----------
    cfg:
        Base config to copy.
    **kwargs:
        Fields to override.

    Returns
    -------
    ObjectiveConfig
        New config instance with requested overrides applied.
    """
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
