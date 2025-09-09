"""Mock hardware interface used for testing.

This module provides a small utility class that mimics the behaviour of the
real DAC/OSA hardware pair.  It allows the optimisation pipeline to run in
environments where no physical devices are available.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence, Tuple, List

import numpy as np

from OCCS.simulate import get_response

logger = logging.getLogger(__name__)


class MockHardware:
    """Simple mock implementation of the hardware interface.

    Parameters
    ----------
    dac_size:
        Number of voltage channels provided by the DAC.
    wavelength:
        Wavelength grid on which the simulated optical response is evaluated.
    """

    def __init__(
        self,
        dac_size: int,
        wavelength: Iterable[float],
        noise_std: Optional[Iterable[float] | float] = None,
        rng: Optional[np.random.Generator] = None,
        voltage_bounds: Optional[Sequence[Tuple[float, float]] | Tuple[float, float]] = None,
    ) -> None:
        """Create a mock hardware instance.

        Parameters
        ----------
        dac_size:
            Number of DAC channels.
        wavelength:
            Sampling grid for the simulated optical response.
        noise_std:
            Optional standard deviation of Gaussian noise added to each
            voltage channel before computing the response.  Can be a scalar or
            broadcastable sequence.  ``None`` disables noise.
        rng:
            Optional random number generator used when ``noise_std`` is not
            ``None``.  If omitted, a default generator is created.
        Additional Parameters
        ---------------------
        voltage_bounds:
            skopt-compatible bounds for each voltage channel. Either a single
            ``(low, high)`` tuple applied to all channels, or a sequence of
            length ``dac_size`` with one ``(low, high)`` per channel. ``None``
            means unbounded.
        """

        self.dac_size = int(dac_size)
        self.wavelength = np.asarray(wavelength, dtype=float)
        self._current_volts = np.zeros(self.dac_size, dtype=float)
        self.noise_std = noise_std
        if noise_std is not None and rng is None:
            rng = np.random.default_rng()
        self._rng = rng
        # Normalise bounds to a list of (low, high) tuples for skopt
        self.voltage_bounds = self._normalise_bounds(voltage_bounds)

    def _normalise_bounds(
        self, bounds: Optional[Sequence[Tuple[float, float]] | Tuple[float, float]]
    ) -> Optional[List[Tuple[float, float]]]:
        if bounds is None:
            return None
        # Single (low, high) pair: broadcast to all channels
        if isinstance(bounds, tuple) and len(bounds) == 2 and not any(
            isinstance(b, (list, tuple)) and len(b) == 2 for b in bounds  # type: ignore[truthy-bool]
        ):
            low, high = float(bounds[0]), float(bounds[1])
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError("Invalid voltage_bounds: expected low < high and both finite")
            return [(low, high) for _ in range(self.dac_size)]

        # Sequence of (low, high)
        try:
            seq = list(bounds)  # type: ignore[arg-type]
        except TypeError as exc:  # not iterable
            raise ValueError("voltage_bounds must be (low, high) or sequence of such") from exc
        if len(seq) != self.dac_size:
            raise ValueError(
                f"voltage_bounds length mismatch: expected {self.dac_size}, got {len(seq)}"
            )
        norm: List[Tuple[float, float]] = []
        for i, pair in enumerate(seq):
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError(f"voltage_bounds[{i}] must be a (low, high) pair")
            lo, hi = float(pair[0]), float(pair[1])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                raise ValueError(f"Invalid bounds at index {i}: low < high and both finite required")
            norm.append((lo, hi))
        return norm

    def apply_voltage(self, new_volts: Iterable[float]) -> None:
        """Update DAC output voltages.

        Parameters
        ----------
        new_volts:
            Sequence of voltages, one per channel.
        """

        volts = np.asarray(new_volts, dtype=float)
        if volts.shape != (self.dac_size,):
            raise ValueError(
                f"Expected {self.dac_size} voltage values, got {volts.shape}"
            )
        # Enforce bounds if provided (clip to valid range)
        if self.voltage_bounds is not None:
            lows = np.array([b[0] for b in self.voltage_bounds], dtype=float)
            highs = np.array([b[1] for b in self.voltage_bounds], dtype=float)
            clipped = np.clip(volts, lows, highs)
            if not np.allclose(clipped, volts):
                logger.debug("Input volts clipped to bounds")
            volts = clipped

        self._current_volts = volts
        logger.debug("Voltage updated: %s", self._current_volts)

    def read_voltage(self) -> np.ndarray:
        """Return the most recently applied voltages."""

        return self._current_volts.copy()

    def get_simulated_response(self) -> np.ndarray:
        """Return the simulated optical response for the current voltages."""

        return get_response(
            self.wavelength,
            self._current_volts,
            noise_std=self.noise_std,
            rng=self._rng,
        )

    # Convenience alias for skopt: dimensions list
    @property
    def skopt_dimensions(self) -> Optional[List[Tuple[float, float]]]:
        return self.voltage_bounds


__all__ = ["MockHardware"]
