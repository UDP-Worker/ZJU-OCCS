"""生成目标方波滤波器响应的工具函数"""

import numpy as np


def create_reference_box_filter(
    frequency_array: np.ndarray,
    center_freq: float,
    fsr: float,
    bandwidth: float,
    passband_level_db: float,
    stopband_level_db: float,
) -> np.ndarray:
    """产生理想方波滤波器幅度响应(dB)."""
    reference_signal = np.full_like(frequency_array, stopband_level_db)
    f_offset = frequency_array - center_freq
    f_normalized = np.mod(f_offset + fsr / 2, fsr) - fsr / 2
    passband_mask = np.abs(f_normalized) <= (bandwidth / 2)
    reference_signal[passband_mask] = passband_level_db
    return reference_signal
