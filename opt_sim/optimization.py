"""基于差分进化的参数优化算法"""

import numpy as np
from scipy.optimize import differential_evolution

from .simulation import optical_simulation


def objective_function(
        params: np.ndarray,
        target_spectrum_db: np.ndarray,
        frequency_f: np.ndarray,
        t: float,
        w_range: np.ndarray,
        H1: np.ndarray,
        H3: np.ndarray,
        n_ku: int,
        m_kl: int,
) -> float:
    """
    自定义损失函数：
    ─ 通带：加权均方误差 (MSE)
    ─ 阻带：对 -20 dB 阈值以上的“超出部分”施加对数障碍惩罚
    """
    # 得到被优化结构的实际响应（dB）
    simulated_db = optical_simulation(params, t, w_range, H1, H3, n_ku, m_kl)

    # 按 “目标响应的最大值” 自动划分通带 / 阻带
    #    这里假设目标在通带全部取同一峰值（典型设置 0 dB）
    pb_level = np.max(target_spectrum_db)
    passband_mask = np.isclose(target_spectrum_db, pb_level, atol=1e-6)
    stopband_mask = ~passband_mask

    # ---------- 通带部分：均方误差 ----------
    mse_passband = np.mean(
        (simulated_db[passband_mask] - target_spectrum_db[passband_mask]) ** 2
    ) if np.any(passband_mask) else 0.0

    # ---------- 阻带部分：对数障碍 ----------
    # 只惩罚高于 -20 dB 的“漏光”点，保留 2 dB 缓冲空间
    leakage_threshold_db = -22.0
    excess = simulated_db[stopband_mask] - leakage_threshold_db        # 正值才违规
    excess_pos = np.maximum(excess, 0.0)                               # 所有负值置零
    barrier = np.log1p(excess_pos) ** 2
    penalty_stopband = np.mean(barrier) if barrier.size else 0.0

    # ③ 组合损失（权重可按需要调整）
    passband_weight = 10.0
    stopband_weight = 1.0
    loss = passband_weight * mse_passband + stopband_weight * penalty_stopband

    return float(loss)

def optimize_params(
    bounds,
    args,
    maxiter: int = 300,
    popsize: int = 20,
) -> differential_evolution:
    """运行差分进化优化器并返回结果."""
    result = differential_evolution(
        objective_function,
        bounds,
        args=args,
        maxiter=maxiter,
        popsize=popsize,
        disp=True,
    )
    return result
