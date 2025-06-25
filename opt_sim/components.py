"""光学芯片中的基础器件模型
=======================

该模块实现可调MZI、相位调制器以及微环谐振器等传输矩阵模型，
供上层仿真与优化算法调用。
"""

import numpy as np


def tunable_mzi_in(theta_i: float) -> np.ndarray:
    """输入端可调MZI矩阵."""
    j = 1j
    coupler_50_50 = 0.5 * np.array([[-1 + j, 1 + j], [1 + j, -1 + j]])
    phase_matrix = np.array([[np.exp(-j * theta_i), 0], [0, 1]])
    return coupler_50_50 @ phase_matrix @ coupler_50_50


def tunable_mzi_out(theta_o: float) -> np.ndarray:
    """输出端可调MZI矩阵."""
    j = 1j
    coupler_50_50 = 0.5 * np.array([[-1 + j, 1 + j], [1 + j, -1 + j]])
    phase_matrix = np.array([[np.exp(-j * theta_o), 0], [0, 1]])
    return coupler_50_50 @ phase_matrix @ coupler_50_50


def phase_shifter_matrix(phi_t: float, phi_b: float) -> np.ndarray:
    """上、下臂相位调制矩阵."""
    j = 1j
    return np.array([[np.exp(-j * phi_t), 0], [0, np.exp(-j * phi_b)]])


def mrr_transfer_function(w: np.ndarray, t: float, k: float, phi_offset: float) -> np.ndarray:
    """微环谐振器 (MRR) 传输函数."""
    j = 1j
    numerator = np.sqrt(1 - k) - t**2 * np.exp(-j * (2 * w + phi_offset))
    denominator = 1 - t**2 * np.sqrt(1 - k) * np.exp(-j * (2 * w + phi_offset))
    eps = 1e-12
    denominator = np.where(np.abs(denominator) < eps, eps + 0j, denominator)
    return numerator / denominator


def delay_line(w: np.ndarray, t: float, delay: float, phi_c: float) -> np.ndarray:
    """简化的延迟线模型."""
    j = 1j
    return t * np.exp(-j * w * delay - j * phi_c)
