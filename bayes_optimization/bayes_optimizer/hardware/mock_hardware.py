"""Mock hardware using simulation."""

import numpy as np
from ..simulate.optical_chip import response as simulate_response


class MockDAC:
    def apply(self, volts: np.ndarray) -> None:
        print(f"[MockDAC] voltages: {volts}")


class MockOSA:
    def read(self, wavelengths: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return simulated spectrum at optional ``wavelengths``."""
        w, resp = simulate_response(MockOSA.current_volts, wavelengths)
        return w, resp

# Store last applied voltages
def apply_voltages(volts: np.ndarray) -> None:
    MockOSA.current_volts = volts

from .. import config

MockOSA.current_volts = np.zeros(config.NUM_CHANNELS)
