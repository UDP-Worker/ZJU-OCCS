"""Utilities for Jacobian measurement and mode compression."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import config
from .hardware import apply, read_spectrum
from .simulate.optical_chip import (
    get_target_waveform,
    compute_loss,
)
from typing import Generator, Dict

# Calibration process should not rely on "ideal voltage" since our final target is to find ideal voltage.
def measure_jacobian(n_samples: int | None = None) -> np.ndarray:
    """Measure sensitivity of the spectrum w.r.t each voltage channel."""
    num_channels = config.NUM_CHANNELS
    # start from mid-range voltages instead of unknown ideal values
    base_volts = np.full(num_channels, sum(config.V_RANGE) / 2)
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


def measure_jacobian_stream() -> Generator[Dict[str, np.ndarray], None, None]:
    """Yield calibration progress for each channel.

    Each yielded dictionary has keys ``step``, ``wavelengths``, ``response``,
    and ``ideal``. The final yield contains ``done`` and ``matrix`` with the
    measured Jacobian.
    """
    num_channels = config.NUM_CHANNELS
    base_volts = np.full(num_channels, sum(config.V_RANGE) / 2)
    wavelengths, ideal = get_target_waveform()
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
        w_p, resp_plus = read_spectrum()
        loss = compute_loss(w_p, resp_plus)
        if ideal.shape != resp_plus.shape or not np.allclose(w_p, wavelengths):
            ideal_for_send = np.interp(w_p, wavelengths, ideal)
        else:
            ideal_for_send = ideal
        apply(v_minus)
        _, resp_minus = read_spectrum()
        J[:, idx] = (resp_plus - resp_minus) / (2 * delta)
        yield {
            "step": idx,
            "wavelengths": w_p,
            "response": resp_plus,
            "ideal": ideal_for_send,
            "base": base_volts[idx],
            "perturb": v_plus[idx],
            "loss": loss,
        }

    yield {"done": True, "matrix": J}


def compress_modes(J: np.ndarray, var_ratio: float = 0.95) -> Tuple[int, np.ndarray]:
    """Compress control channels using PCA."""
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    explained = np.cumsum(S ** 2) / np.sum(S ** 2)
    n_components = int(np.searchsorted(explained, var_ratio) + 1)
    compression = Vt[:n_components].T
    return n_components, compression
