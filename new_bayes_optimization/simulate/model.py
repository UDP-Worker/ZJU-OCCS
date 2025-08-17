"""Lightweight optical chip simulation.

The simulation is intentionally simple: the optical response is modelled as a
linear combination of cosine modes whose amplitudes are controlled by the input
voltages.  This provides a deterministic environment for testing the Bayesian
optimisation logic without access to real hardware.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def get_response(
    wavelength: Iterable[float],
    input_volts: Iterable[float],
    noise_std: Optional[Iterable[float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute the simulated optical response.

    Parameters
    ----------
    wavelength:
        Sequence of wavelength samples.
    input_volts:
        Control voltages for each mode.
    noise_std:
        Optional standard deviation of Gaussian noise applied independently to
        each voltage.  Can be a scalar or sequence matching ``input_volts``.
        If ``None`` (default) no noise is added.
    rng:
        Optional ``numpy.random.Generator`` instance used to draw the noise.
        If omitted, ``np.random.default_rng()`` is used when ``noise_std`` is
        provided.

    Returns
    -------
    np.ndarray
        Simulated response at each wavelength.
    """

    wavelength = np.asarray(wavelength, dtype=float)
    input_volts = np.asarray(input_volts, dtype=float)

    if noise_std is not None:
        noise_std = np.asarray(noise_std, dtype=float)
        try:
            noise_std = np.broadcast_to(noise_std, input_volts.shape)
        except ValueError as exc:
            raise ValueError(
                "noise_std must be broadcastable to the shape of input_volts"
            ) from exc
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(0.0, noise_std, size=input_volts.shape)
        input_volts = input_volts + noise

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

