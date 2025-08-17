"""Placeholder for real hardware interaction.

The project currently uses :class:`~new_bayes_optimization.connector.MockHardware`
for all simulations and tests.  This module defines a minimal stub that mirrors
the public API and makes it clear that the real hardware backend still needs to
be implemented.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


class RealHardware:
    """Interface to the actual DAC/OSA hardware.

    The implementation is intentionally left blank; calling any method will
    raise :class:`NotImplementedError`.  Replace the stubs with the actual
    device communication code when integrating with hardware.
    """

    def __init__(self, dac_size: int, wavelength: Iterable[float]) -> None:
        self.dac_size = int(dac_size)
        self.wavelength = np.asarray(wavelength, dtype=float)
        raise NotImplementedError("Real hardware driver is not implemented")

    def apply_voltage(self, new_volts: Iterable[float]) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    def read_voltage(self) -> np.ndarray:  # pragma: no cover - placeholder
        raise NotImplementedError

    def get_response(self) -> np.ndarray:  # pragma: no cover - placeholder
        raise NotImplementedError


__all__ = ["RealHardware"]

