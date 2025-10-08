import os

import numpy as np
import pytest

from OCCS.connector.real_hardware import RealHardware, HardwareUnavailableError
from OCCS.connector.hardware_config import load_real_hardware_config, DEFAULT_CONFIG_PATH


def _config_for_channels(channels):
    return {
        "enabled": True,
        "bounds": [0.0, 2.0],
        "dac": {
            "port": "COM_TEST",
            "channels": list(channels),
            "baudrate": 115200,
            "timeout": 1.0,
            "write_timeout": 1.0,
            "query_delay": 0.0,
            "read_termination": "\n",
            "write_termination": "\n",
        },
        "osa": {
            "resource": "GPIB::TEST",
            "channel": "a",
            "sensitivity": "high2",
            "speed": "2x",
            "timeout": 10.0,
            "chunk_size": 1000,
        },
    }


class DummyDAC:
    def __init__(self):
        self.commands = {}
        self.vmax = {}
        self.closed = False

    def set_voltage(self, channel: int, value: float) -> None:
        self.commands[int(channel)] = float(value)

    def get_voltage(self, channel: int) -> float:
        return float(self.commands.get(int(channel), 0.0))

    def set_vmax(self, channel: int, value: float) -> None:
        self.vmax[int(channel)] = float(value)

    def close(self) -> None:
        self.closed = True


class DummyDACWithFailure(DummyDAC):
    def get_voltage(self, channel: int) -> float:
        raise HardwareUnavailableError("no readback")


class DummyOSA:
    def __init__(self, lam_nm: np.ndarray, power: np.ndarray):
        self.lam_nm = lam_nm
        self.power = power
        self.calls = []
        self.closed = False

    def acquire_trace(self, start_nm: float, stop_nm: float, points: int, resolution_nm: float):
        self.calls.append((start_nm, stop_nm, points, resolution_nm))
        return self.lam_nm, self.power

    def close(self) -> None:
        self.closed = True


def _make_wavelength(size: int = 5) -> np.ndarray:
    return np.linspace(1.55e-6, 1.56e-6, size)


def test_apply_and_read_voltage_with_stubs():
    lam = _make_wavelength()
    dac = DummyDAC()
    osa = DummyOSA(lam * 1e9, np.linspace(-10.0, -5.0, lam.size))
    hw = RealHardware(
        3,
        lam,
        channel_ids=[31, 30, 29],
        dac_controller=dac,
        osa_controller=osa,
        config=_config_for_channels([31, 30, 29]),
    )

    hw.apply_voltage([0.1, 0.2, 0.3])
    assert dac.commands == {31: 0.1, 30: 0.2, 29: 0.3}
    assert dac.vmax == {31: 2.0, 30: 2.0, 29: 2.0}

    readback = hw.read_voltage()
    assert np.allclose(readback, [0.1, 0.2, 0.3])

    response = hw.get_response()
    assert response.shape == (lam.size,)
    assert np.allclose(response, np.linspace(-10.0, -5.0, lam.size))

    hw.close()
    assert dac.closed is True
    assert osa.closed is True


def test_read_voltage_falls_back_to_cache():
    lam = _make_wavelength()
    dac = DummyDACWithFailure()
    osa = DummyOSA(lam * 1e9, np.zeros(lam.size))
    hw = RealHardware(
        2,
        lam,
        voltage_bounds=[(-1.0, 1.0)] * 2,
        dac_controller=dac,
        osa_controller=osa,
        config=_config_for_channels([1, 2]),
    )

    hw.apply_voltage([0.4, -0.2])
    readback = hw.read_voltage()
    assert np.allclose(readback, [0.4, -0.2])


def test_response_interpolation_applied_when_grid_differs():
    lam = _make_wavelength()
    lam_instr = np.linspace(lam[0] * 1e9, lam[-1] * 1e9, lam.size + 2)
    power_instr = np.linspace(-15.0, -5.0, lam_instr.size)
    dac = DummyDAC()
    osa = DummyOSA(lam_instr, power_instr)
    hw = RealHardware(
        4,
        lam,
        dac_controller=dac,
        osa_controller=osa,
        config=_config_for_channels([1, 2, 3, 4]),
    )

    hw.apply_voltage([0.0] * 4)
    response = hw.get_response()
    assert response.shape == (lam.size,)
    expected = np.interp(lam, lam_instr * 1e-9, power_instr)
    assert np.allclose(response, expected)


@pytest.mark.hardware
def test_live_hardware_optional():
    if os.environ.get("OCCS_TEST_REAL_HARDWARE") != "1":
        pytest.skip("Live hardware test disabled")

    config = load_real_hardware_config(DEFAULT_CONFIG_PATH)
    if not config:
        pytest.skip("hardware configuration not set")
    channels = config.get("dac", {}).get("channels")
    dac_size = len(channels) if channels else 1
    lam = _make_wavelength(8)
    try:
        hw = RealHardware(
            dac_size,
            lam,
        )
    except HardwareUnavailableError as exc:
        pytest.skip(f"Hardware unavailable: {exc}")
    try:
        hw.apply_voltage([0.0])
        response = hw.get_response()
        assert response.shape == (lam.size,)
    finally:
        hw.close()
