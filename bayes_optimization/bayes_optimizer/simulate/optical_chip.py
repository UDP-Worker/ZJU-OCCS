"""Parametric optical chip simulation used for testing and the mock UI."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import csv

DATA_FILE = Path(__file__).with_name("ideal_waveform.csv")

# Default wavelengths used when the caller does not specify one.
_DEFAULT_WAVELENGTHS = np.linspace(1.55e-6, 1.56e-6, 200)

# Target waveform that optimisation tries to match.
_target_wl: np.ndarray
_target_resp: np.ndarray

# Cached per-(channels, points) basis arrays and channel mixing matrices.
_basis_cache: dict[tuple[int, int], np.ndarray] = {}
_mix_cache: dict[int, np.ndarray] = {}

# Ideal voltages are only used as a starting guess for the optimiser.
_ideal_voltages: np.ndarray | None = None


def _default_ideal_voltages(n: int) -> np.ndarray:
    """Return a fixed non-uniform voltage pattern for realism."""
    return np.linspace(0.4, 1.6, n)


def _load_waveform(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    wl = np.asarray(rows[0], dtype=float)
    resp = np.asarray(rows[1], dtype=float)
    return wl, resp


_target_wl, _target_resp = _load_waveform(DATA_FILE)


def set_target_waveform(wavelengths: np.ndarray, response: np.ndarray, ideal_voltages: np.ndarray | None = None) -> None:
    """Update the target waveform used for optimisation."""
    global _target_wl, _target_resp, _ideal_voltages
    _target_wl = np.asarray(wavelengths, dtype=float)
    _target_resp = np.asarray(response, dtype=float)
    if ideal_voltages is not None:
        _ideal_voltages = np.asarray(ideal_voltages, dtype=float)
    else:
        _ideal_voltages = None


def get_target_waveform() -> tuple[np.ndarray, np.ndarray]:
    return _target_wl, _target_resp


def get_ideal_voltages(num_channels: int) -> np.ndarray:
    global _ideal_voltages
    if _ideal_voltages is None or len(_ideal_voltages) != num_channels:
        _ideal_voltages = _default_ideal_voltages(num_channels)
    return _ideal_voltages


def _get_basis(num_channels: int, n_points: int) -> np.ndarray:
    key = (num_channels, n_points)
    basis = _basis_cache.get(key)
    if basis is None:
        x = np.linspace(0, np.pi, n_points)
        basis = np.array([np.sin((i + 1) * x) for i in range(num_channels)]) / np.sqrt(num_channels)
        _basis_cache[key] = basis
    return basis


def _get_mix(num_channels: int) -> np.ndarray:
    mix = _mix_cache.get(num_channels)
    if mix is None:
        rng = np.random.default_rng(0)
        mix = rng.normal(0.0, 0.5, size=(num_channels, num_channels))
        _mix_cache[num_channels] = mix
    return mix


def response(volts: np.ndarray, wavelengths: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return simulated spectrum for the given voltages."""
    if wavelengths is None:
        w = _DEFAULT_WAVELENGTHS
    else:
        w = np.asarray(wavelengths, dtype=float)
    num_channels = len(volts)
    basis = _get_basis(num_channels, w.size)
    mix = _get_mix(num_channels)
    ideal = get_ideal_voltages(num_channels)
    diff = volts - ideal
    patterns = mix @ basis
    delta = (diff @ patterns) / num_channels
    base = np.full_like(w, -30.0)
    simulated = base + delta
    return w.copy(), simulated


def compute_loss(wavelengths: np.ndarray, response: np.ndarray) -> float:
    """Mean squared error against the current target waveform."""
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
