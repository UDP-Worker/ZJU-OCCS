"""Simple optical chip response simulation."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import csv

DATA_FILE = Path(__file__).with_name("ideal_waveform.csv")


def _default_ideal_voltages(n: int) -> np.ndarray:
    """Return a fixed non-uniform voltage pattern for realism."""
    return np.linspace(0.4, 1.6, n)


def _load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load ideal waveform from CSV."""
    with open(DATA_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    wavelengths = np.array(rows[0], dtype=float)
    response = np.array(rows[1], dtype=float)
    return wavelengths, response


_BASE_WAVELENGTHS, _BASE_RESPONSE = _load_data()
# Target waveform user wants to match
_TARGET_WAVELENGTHS = _BASE_WAVELENGTHS.copy()
_TARGET_RESPONSE = _BASE_RESPONSE.copy()
# optimal voltages corresponding to the ideal waveform
_IDEAL_VOLTAGES: np.ndarray | None = None
_BASIS: np.ndarray | None = None
_MIX: np.ndarray | None = None


def set_target_waveform(
    wavelengths: np.ndarray,
    response: np.ndarray,
    ideal_voltages: np.ndarray | None = None,
) -> None:
    """Update the target waveform used for optimization."""
    global _TARGET_WAVELENGTHS, _TARGET_RESPONSE, _IDEAL_VOLTAGES
    _TARGET_WAVELENGTHS = np.asarray(wavelengths, dtype=float)
    _TARGET_RESPONSE = np.asarray(response, dtype=float)
    if ideal_voltages is not None:
        _IDEAL_VOLTAGES = np.asarray(ideal_voltages, dtype=float)
    else:
        _IDEAL_VOLTAGES = None


def get_target_waveform() -> tuple[np.ndarray, np.ndarray]:
    """Return current target waveform."""
    return _TARGET_WAVELENGTHS, _TARGET_RESPONSE


def get_ideal_voltages(num_channels: int) -> np.ndarray:
    """Return the baseline voltages for the given channel count."""
    global _IDEAL_VOLTAGES
    if _IDEAL_VOLTAGES is None or len(_IDEAL_VOLTAGES) != num_channels:
        _IDEAL_VOLTAGES = _default_ideal_voltages(num_channels)
    return _IDEAL_VOLTAGES


def response(volts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return simulated spectrum for given voltages."""
    num_channels = len(volts)
    n = len(_BASE_RESPONSE)
    global _BASIS, _MIX

    ideal = get_ideal_voltages(num_channels)

    if _BASIS is None or _BASIS.shape != (num_channels, n):
        x = np.linspace(0, np.pi, n)
        _BASIS = np.array([np.sin((i + 1) * x) for i in range(num_channels)]) / np.sqrt(
            num_channels
        )

    if _MIX is None or _MIX.shape != (num_channels, num_channels):
        rng = np.random.default_rng(0)
        _MIX = rng.normal(0.0, 0.5, size=(num_channels, num_channels))

    diff = volts - ideal
    patterns = _MIX @ _BASIS
    delta = (diff @ patterns) / num_channels

    # amplify influence of voltages so manual adjustment has visible effect
    simulated = _BASE_RESPONSE + 1.0 * delta
    return _BASE_WAVELENGTHS.copy(), simulated


def compute_loss(
    wavelengths: np.ndarray, response: np.ndarray
) -> float:
    """Return mean squared error against the current target waveform.

    If the provided wavelengths do not match the target waveform, the target is
    interpolated accordingly so that arbitrary uploaded targets are supported
    without dimension mismatch errors.
    """
    target_wl, target_resp = get_target_waveform()
    if (
        response.shape != target_resp.shape
        or wavelengths.shape != target_wl.shape
        or not np.allclose(wavelengths, target_wl)
    ):
        target_interp = np.interp(wavelengths, target_wl, target_resp)
    else:
        target_interp = target_resp
    return float(np.mean((response - target_interp) ** 2))
