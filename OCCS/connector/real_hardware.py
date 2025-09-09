"""Placeholder for real hardware interaction.

The project currently uses :class:`~OCCS.connector.MockHardware`
for all simulations and tests.  This module defines a minimal stub that mirrors
the public API and makes it clear that the real hardware backend still needs to
be implemented.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, List

import numpy as np


class RealHardware:
    """Interface to the actual DAC/OSA hardware.

    The implementation is intentionally left blank; calling any method will
    raise :class:`NotImplementedError`.  Replace the stubs with the actual
    device communication code when integrating with hardware.
    """

    def __init__(
        self,
        dac_size: int,
        wavelength: Iterable[float],
        voltage_bounds: Optional[Sequence[Tuple[float, float]] | Tuple[float, float]] = None,
    ) -> None:
        self.dac_size = int(dac_size)
        self.wavelength = np.asarray(wavelength, dtype=float)
        # Store skopt-compatible bounds (list of (low, high) or None)
        self.voltage_bounds: Optional[List[Tuple[float, float]]] = (
            list(voltage_bounds)  # type: ignore[list-item]
            if isinstance(voltage_bounds, (list, tuple)) else None
        )
        raise NotImplementedError("Real hardware driver is not implemented")

    def apply_voltage(self, new_volts: Iterable[float]) -> None:  # pragma: no cover - placeholder
        """Apply a voltage vector to the DAC.

        Parameters
        ----------
        new_volts:
            Sequence of voltages, one per channel.
        """
        raise NotImplementedError

    def read_voltage(self) -> np.ndarray:  # pragma: no cover - placeholder
        """Return the most recently applied voltages as an array."""
        raise NotImplementedError

    def get_response(self) -> np.ndarray:  # pragma: no cover - placeholder
        """Acquire the current optical response from the OSA."""
        raise NotImplementedError

    @property
    def skopt_dimensions(self):  # pragma: no cover - placeholder
        """Bounds formatted for scikit-optimize (list of ``(low, high)``)."""
        return self.voltage_bounds


__all__ = ["RealHardware"]
