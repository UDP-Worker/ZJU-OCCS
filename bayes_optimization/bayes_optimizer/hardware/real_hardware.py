import numpy as np

class RealDAC:
    def __init__(self):
        raise ConnectionError("Real DAC not available")

    def apply(self, volts: np.ndarray) -> None:
        pass


class RealOSA:
    def __init__(self):
        raise ConnectionError("Real OSA not available")

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([]), np.array([])
