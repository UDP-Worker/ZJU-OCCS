"""Mock hardware interface used for testing.

This module provides a small utility class that mimics the behaviour of the
real DAC/OSA hardware pair.  It allows the optimisation pipeline to run in
environments where no physical devices are available.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np

from new_bayes_optimization.simulate import get_response

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
        """

        self.dac_size = int(dac_size)
        self.wavelength = np.asarray(wavelength, dtype=float)
        self._current_volts = np.zeros(self.dac_size, dtype=float)
        self.noise_std = noise_std
        if noise_std is not None and rng is None:
            rng = np.random.default_rng()
        self._rng = rng

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


__all__ = ["MockHardware"]

