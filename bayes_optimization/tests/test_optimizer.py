import numpy as np
from bayes_optimization.bayes_optimizer.models import GaussianProcess
from bayes_optimization.bayes_optimizer.optimizer import BayesOptimizer
from bayes_optimization.bayes_optimizer.acquisition import expected_improvement
from bayes_optimization.bayes_optimizer.spsa import spsa_refine
from bayes_optimization.bayes_optimizer.simulate.optical_chip import (
    response,
    _TARGET_RESPONSE,
    get_ideal_voltages,
    compute_loss,
)


def loss_fn(volts: np.ndarray) -> float:
    w, resp = response(volts)
    return compute_loss(w, resp)


def test_bo_spsa_converges():
    """End-to-end optimization on a 5-channel system."""
    num_ch = 5
    bounds = np.tile([[0.0, 2.0]], (num_ch, 1))
    start = np.full(num_ch, 1.0)
    bo = BayesOptimizer(GaussianProcess(), expected_improvement, bounds)
    res = bo.optimize(start, loss_fn, steps=10)
    refined = spsa_refine(res["best_x"], loss_fn, a0=0.5, c0=0.1, steps=50)
    final_loss = loss_fn(refined)
    assert final_loss < 0.02
    assert not np.allclose(refined, np.ones(num_ch))


def test_large_channel_optimization():
    """Optimization should still work with many channels."""
    num_ch = 32
    bounds = np.tile([[0.0, 2.0]], (num_ch, 1))
    start = np.zeros(num_ch)
    bo = BayesOptimizer(GaussianProcess(), expected_improvement, bounds)
    res = bo.optimize(start, loss_fn, steps=15)
    refined = spsa_refine(res["best_x"], loss_fn, a0=0.5, c0=0.1, steps=80)
    final_loss = loss_fn(refined)
    assert final_loss < 0.2
    assert not np.allclose(refined, np.ones(num_ch))
