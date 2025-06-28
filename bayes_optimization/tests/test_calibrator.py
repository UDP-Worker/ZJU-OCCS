import numpy as np
from bayes_optimization.bayes_optimizer import config
from bayes_optimization.bayes_optimizer.calibrator import (
    measure_jacobian,
    compress_modes,
)
from bayes_optimization.bayes_optimizer.simulate.optical_chip import (
    _IDEAL_RESPONSE,
)


def test_measure_jacobian_shape():
    J = measure_jacobian()
    assert J.shape == (_IDEAL_RESPONSE.size, config.NUM_CHANNELS)
    assert np.all(np.isfinite(J))


def test_compress_modes():
    J = measure_jacobian()
    n, mat = compress_modes(J, var_ratio=0.9)
    assert 1 <= n <= config.NUM_CHANNELS
    assert mat.shape == (config.NUM_CHANNELS, n)
    # Components should be orthonormal
    prod = mat.T @ mat
    assert np.allclose(prod, np.eye(n), atol=1e-6)
