"""Simple hardware abstraction switching between mock and real devices."""

from __future__ import annotations

import numpy as np

from . import mock_hardware

MODE = "mock"
CONNECTED = True
_dac: any = mock_hardware.MockDAC()
_osa: any = mock_hardware.MockOSA()


def set_mode(mode: str) -> bool:
    """Switch hardware backend. Returns True if connected."""
    global MODE, _dac, _osa, CONNECTED
    MODE = mode
    if mode == "real":
        try:
            from .real_hardware import RealDAC, RealOSA
            _dac = RealDAC()
            _osa = RealOSA()
            CONNECTED = True
        except Exception:
            _dac = None
            _osa = None
            CONNECTED = False
    else:
        _dac = mock_hardware.MockDAC()
        _osa = mock_hardware.MockOSA()
        CONNECTED = True
    return CONNECTED


def apply(volts: np.ndarray) -> None:
    """Apply voltages to DAC."""
    if _dac is None:
        raise ConnectionError("hardware not connected")
    if MODE == "mock":
        mock_hardware.apply_voltages(volts)
    _dac.apply(volts)


def read_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Read spectrum from OSA."""
    if _osa is None:
        raise ConnectionError("hardware not connected")
    return _osa.read()
