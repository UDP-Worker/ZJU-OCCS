import numpy as np
from bayes_optimization.bayes_optimizer import config
from bayes_optimization.bayes_optimizer.calibrator import (
    measure_jacobian,
    compress_modes,
)
from bayes_optimization.bayes_optimizer.simulate import optical_chip


def test_measure_jacobian_shape():
    target_wl, target_resp = optical_chip.get_target_waveform()
    J = measure_jacobian()
    assert J.shape == (target_resp.size, config.NUM_CHANNELS)
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

    base = optical_chip.get_ideal_voltages(config.NUM_CHANNELS)
    perturbed = base.copy()
    perturbed[0] += 0.01
    target_wl, _ = optical_chip.get_target_waveform()
    w, resp = optical_chip.response(perturbed, target_wl)
    loss = optical_chip.compute_loss(w, resp)
    assert loss > 1e-8
