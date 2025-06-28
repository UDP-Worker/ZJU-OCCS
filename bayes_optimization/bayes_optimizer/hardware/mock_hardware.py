"""Mock hardware using simulation."""

import numpy as np
from ..simulate.optical_chip import response as simulate_response


class MockDAC:
    def apply(self, volts: np.ndarray) -> None:
        print(f"[MockDAC] voltages: {volts}")


class MockOSA:
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        w, resp = simulate_response(MockOSA.current_volts)
        return w, resp

# Store last applied voltages
def apply_voltages(volts: np.ndarray) -> None:
    MockOSA.current_volts = volts

MockOSA.current_volts = np.zeros(1)
