import numpy as np

from new_bayes_optimization.connector import MockHardware
from new_bayes_optimization.simulate import get_response


def test_get_response_shape_and_linearity():
    wavelengths = np.linspace(0.0, 1.0, 10)
    volts1 = np.array([1.0, 0.0, 0.0])
    volts2 = np.array([0.0, 1.0, 0.0])

    resp1 = get_response(wavelengths, volts1)
    resp2 = get_response(wavelengths, volts2)
    resp12 = get_response(wavelengths, volts1 + volts2)

    assert resp1.shape == wavelengths.shape
    # Linearity check
    np.testing.assert_allclose(resp1 + resp2, resp12)


def test_mock_hardware_interface():
    wavelengths = np.linspace(0.0, 1.0, 20)
    hardware = MockHardware(3, wavelengths, noise_std=0.0)
    volts = np.array([0.1, 0.2, 0.3])

    hardware.apply_voltage(volts)
    np.testing.assert_allclose(hardware.read_voltage(), volts)
    resp = hardware.get_simulated_response()
    assert resp.shape == wavelengths.shape


def test_mock_hardware_noise_injection():
    wavelengths = np.linspace(0.0, 1.0, 20)
    volts = np.array([0.1, 0.2, 0.3])
    hardware = MockHardware(3, wavelengths, noise_std=0.1, rng=np.random.default_rng(0))
    hardware.apply_voltage(volts)
    resp1 = hardware.get_simulated_response()
    resp2 = hardware.get_simulated_response()
    # successive reads should differ due to injected noise
    assert not np.allclose(resp1, resp2)

