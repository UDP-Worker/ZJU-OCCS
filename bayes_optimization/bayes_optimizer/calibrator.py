"""Utilities for Jacobian measurement and mode compression."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import config
from .hardware import apply, read_spectrum


def measure_jacobian(n_samples: int | None = None) -> np.ndarray:
    """Measure sensitivity of the spectrum w.r.t each voltage channel."""
    num_channels = config.NUM_CHANNELS
    base_volts = np.zeros(num_channels)
    apply(base_volts)
    _, base_resp = read_spectrum()
    num_feat = base_resp.size
    J = np.zeros((num_feat, num_channels))
    delta = 1e-2

    for idx in range(num_channels):
        v_plus = base_volts.copy()
        v_minus = base_volts.copy()
        v_plus[idx] += delta
        v_minus[idx] -= delta
        apply(v_plus)
        _, resp_plus = read_spectrum()
        apply(v_minus)
        _, resp_minus = read_spectrum()
        J[:, idx] = (resp_plus - resp_minus) / (2 * delta)

    return J


def compress_modes(J: np.ndarray, var_ratio: float = 0.95) -> Tuple[int, np.ndarray]:
    """Compress control channels using PCA."""
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    explained = np.cumsum(S ** 2) / np.sum(S ** 2)
    n_components = int(np.searchsorted(explained, var_ratio) + 1)
    compression = Vt[:n_components].T
    return n_components, compression
