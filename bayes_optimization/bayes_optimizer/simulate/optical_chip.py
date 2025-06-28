"""Simple optical chip response simulation."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import csv

DATA_FILE = Path(__file__).with_name("ideal_waveform.csv")


def _load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load ideal waveform from CSV."""
    with open(DATA_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    wavelengths = np.array(rows[0], dtype=float)
    response = np.array(rows[1], dtype=float)
    return wavelengths, response


_WAVELENGTHS, _IDEAL_RESPONSE = _load_data()
# optimal voltages corresponding to the ideal waveform; by default all ones
_IDEAL_VOLTAGES: np.ndarray | None = None


def set_target_waveform(
    wavelengths: np.ndarray,
    response: np.ndarray,
    ideal_voltages: np.ndarray | None = None,
) -> None:
    """Update the target waveform used for optimization."""
    global _WAVELENGTHS, _IDEAL_RESPONSE, _IDEAL_VOLTAGES
    _WAVELENGTHS = np.asarray(wavelengths, dtype=float)
    _IDEAL_RESPONSE = np.asarray(response, dtype=float)
    if ideal_voltages is not None:
        _IDEAL_VOLTAGES = np.asarray(ideal_voltages, dtype=float)


def get_target_waveform() -> tuple[np.ndarray, np.ndarray]:
    """Return current target waveform."""
    return _WAVELENGTHS, _IDEAL_RESPONSE


def response(volts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return simulated spectrum for given voltages."""
    num_channels = len(volts)
    n = len(_IDEAL_RESPONSE)
    global _IDEAL_VOLTAGES
    if _IDEAL_VOLTAGES is None or len(_IDEAL_VOLTAGES) != num_channels:
        _IDEAL_VOLTAGES = np.ones(num_channels)

    patterns = np.array(
        [np.sin((i + 1) * np.linspace(0, np.pi, n)) for i in range(num_channels)]
    ) / np.sqrt(num_channels)

    delta = (volts - _IDEAL_VOLTAGES) @ patterns
    # amplify influence of voltages so manual adjustment has visible effect
    simulated = _IDEAL_RESPONSE + 0.15 * delta
    return _WAVELENGTHS.copy(), simulated
