import numpy as np

from OCCS.service.hardware import list_backends, make_hardware


def test_list_backends_contains_mock():
    backends = list_backends()
    names = {b["name"] for b in backends}
    assert "mock" in names


def test_make_hardware_mock_basic_response_shape():
    lam = np.linspace(1.55e-6, 1.56e-6, 128)
    hw = make_hardware(
        "mock",
        dac_size=3,
        wavelength=lam,
        bounds=[(-1.0, 1.0)] * 3,
    )
    # Apply zeros and read a response
    hw.apply_voltage([0.0, 0.0, 0.0])
    y = hw.get_response()
    assert isinstance(y, np.ndarray)
    assert y.shape == (lam.size,)


def test_make_hardware_real_unavailable():
    lam = np.linspace(1.55e-6, 1.56e-6, 32)
    try:
        make_hardware("real", dac_size=1, wavelength=lam)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Expected NotImplementedError for real backend by default")

