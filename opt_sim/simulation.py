"""滤波器传输矩阵仿真"""

import numpy as np

from .components import (
    tunable_mzi_in,
    tunable_mzi_out,
    phase_shifter_matrix,
    mrr_transfer_function,
    delay_line,
)


def optical_simulation(
    params: np.ndarray,
    t: float,
    w_range: np.ndarray,
    H1: np.ndarray,
    H3: np.ndarray,
    n_ku: int,
    m_kl: int,
) -> np.ndarray:
    """根据给定参数计算 |H11| 的幅度(dB)."""
    ku_params = params[:n_ku]
    kl_params = params[n_ku:]

    len_w = len(w_range)

    Au_mrr_responses = [mrr_transfer_function(w_range, t, k, phi_offset=np.pi) for k in ku_params]
    Au = np.prod(Au_mrr_responses, axis=0) if len(Au_mrr_responses) > 0 else 1.0

    if m_kl > 0:
        Al_mrr_responses = [mrr_transfer_function(w_range, t, k, phi_offset=np.pi) for k in kl_params]
        Al_mrr_product = np.prod(Al_mrr_responses, axis=0)
    else:
        Al_mrr_product = 1.0

    Al = Al_mrr_product * delay_line(w_range, t, delay=1.0, phi_c=0.0)

    H2_stack = np.zeros((len_w, 2, 2), dtype=complex)
    H2_stack[:, 0, 0] = Au
    H2_stack[:, 1, 1] = Al

    H_final = H1 @ H2_stack @ H3
    H11 = H_final[:, 0, 0]

    return 20 * np.log10(np.abs(H11))


__all__ = [
    "tunable_mzi_in",
    "tunable_mzi_out",
    "phase_shifter_matrix",
    "optical_simulation",
]
