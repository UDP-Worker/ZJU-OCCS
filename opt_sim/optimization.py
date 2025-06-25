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
    """加权均方误差损失函数."""
    simulated = optical_simulation(params, t, w_range, H1, H3, n_ku, m_kl)

    passband_weight = 100.0
    stopband_weight = 1.0
    weights = np.full_like(frequency_f, stopband_weight)
    passband_indices = np.where(target_spectrum_db == 0)
    weights[passband_indices] = passband_weight

    return float(np.mean(weights * (simulated - target_spectrum_db) ** 2))


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
