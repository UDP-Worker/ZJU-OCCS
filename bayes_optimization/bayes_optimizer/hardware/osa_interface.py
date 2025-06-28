"""OSA interface placeholder."""

import numpy as np
from .mock_hardware import MockOSA

_osa = MockOSA()


def read_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Return wavelength and response arrays."""
    return _osa.read()
