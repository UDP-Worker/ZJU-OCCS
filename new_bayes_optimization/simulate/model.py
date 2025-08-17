"""Lightweight optical chip simulation.

The simulation is intentionally simple: the optical response is modelled as a
linear combination of cosine modes whose amplitudes are controlled by the input
voltages.  This provides a deterministic environment for testing the Bayesian
optimisation logic without access to real hardware.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def get_response(wavelength: Iterable[float], input_volts: Iterable[float]) -> np.ndarray:
    """Compute the simulated optical response.

    Parameters
    ----------
    wavelength:
        Sequence of wavelength samples.
    input_volts:
        Control voltages for each mode.

    Returns
    -------
    np.ndarray
        Simulated response at each wavelength.
    """

    wavelength = np.asarray(wavelength, dtype=float)
    input_volts = np.asarray(input_volts, dtype=float)

    w_min, w_max = wavelength.min(), wavelength.max()
    if w_max == w_min:
        raise ValueError("wavelength 所有元素相同，无法做线性映射。")

    # Map wavelengths to [-pi, pi] for mode computation
    x = (wavelength - w_min) / (w_max - w_min) * (2 * np.pi) - np.pi

    # Construct cosine series
    i = np.arange(1, input_volts.size + 1, dtype=float)[:, None]
    basis = np.cos(i * x)
    response = (input_volts[:, None] * basis).sum(axis=0)
    return response


__all__ = ["get_response"]

