import numpy as np
import pytest

from OCCS.service.models import WavelengthSpec, normalise_bounds


def test_wavelengthspec_array_roundtrip():
    arr = [1.0, 2.0, 3.0]
    spec = WavelengthSpec(array=arr)
    out = spec.resolve()
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.allclose(out, np.array(arr, dtype=float))


def test_wavelengthspec_linspace_generation():
    spec = WavelengthSpec(start=0.0, stop=1.0, M=5)
    out = spec.resolve()
    assert out.shape == (5,)
    assert np.isclose(out[0], 0.0)
    assert np.isclose(out[-1], 1.0)


def test_normalise_bounds_broadcast():
    out = normalise_bounds((-1.0, 1.0), dac_size=3)
    assert out == [(-1.0, 1.0)] * 3


def test_normalise_bounds_sequence_and_validation():
    with pytest.raises(ValueError):
        normalise_bounds([(0.0, 1.0)], dac_size=2)  # length mismatch
    with pytest.raises(ValueError):
        normalise_bounds([(1.0, 0.0)], dac_size=1)  # low >= high
    out = normalise_bounds([(0.0, 1.0), (-2.0, 2.0)], dac_size=2)
    assert out == [(0.0, 1.0), (-2.0, 2.0)]

