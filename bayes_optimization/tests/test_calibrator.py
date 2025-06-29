import numpy as np
from bayes_optimization.bayes_optimizer import config
from bayes_optimization.bayes_optimizer.calibrator import (
    measure_jacobian,
    compress_modes,
)
from bayes_optimization.bayes_optimizer.simulate.optical_chip import (
    _IDEAL_RESPONSE,
    get_ideal_voltages,
    response,
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


def test_calibration_effect():
    """Perturbing a channel should change the loss and Jacobian."""
    J = measure_jacobian()
    # matrix should not collapse to identical rows
    assert np.std(J) > 0.0

    base = get_ideal_voltages(config.NUM_CHANNELS)
    perturbed = base.copy()
    perturbed[0] += 0.01
    _, resp = response(perturbed)
    loss = float(np.mean((resp - _IDEAL_RESPONSE) ** 2))
    assert loss > 1e-8
