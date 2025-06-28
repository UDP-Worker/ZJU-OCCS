"""Utilities for Jacobian measurement and mode compression."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import config
from .simulate import response


def measure_jacobian(n_samples: int | None = None) -> np.ndarray:
    """Measure sensitivity of the spectrum w.r.t each voltage channel."""
    num_channels = config.NUM_CHANNELS
    base_volts = np.zeros(num_channels)
    _, base_resp = response(base_volts)
    num_feat = base_resp.size
    J = np.zeros((num_feat, num_channels))
    delta = 1e-2

    for idx in range(num_channels):
        v_plus = base_volts.copy()
        v_minus = base_volts.copy()
        v_plus[idx] += delta
        v_minus[idx] -= delta
        _, resp_plus = response(v_plus)
        _, resp_minus = response(v_minus)
        J[:, idx] = (resp_plus - resp_minus) / (2 * delta)

    return J


def compress_modes(J: np.ndarray, var_ratio: float = 0.95) -> Tuple[int, np.ndarray]:
    """Compress control channels using PCA."""
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    explained = np.cumsum(S ** 2) / np.sum(S ** 2)
    n_components = int(np.searchsorted(explained, var_ratio) + 1)
    compression = Vt[:n_components].T
    return n_components, compression
