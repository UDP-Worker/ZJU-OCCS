import os
import sys
import numpy as np
import pytest

# Ensure package import from repo root
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from new_bayes_optimization.connector import MockHardware
from new_bayes_optimization.simulate import get_response
from new_bayes_optimization.optimizer.objective import CurveObjective, HardwareObjective
from new_bayes_optimization.optimizer.optimizer import BayesianOptimizer


def test_skopt_optimizes_within_bounds_and_reduces_loss():
    # Deterministic setup
    rng = np.random.default_rng(123)
    lam = np.linspace(1.55e-6, 1.56e-6, 200)

    # True parameters to recover (3 channels)
    true_volts = np.array([0.6, -0.25, 0.9], dtype=float)
    bounds = [(-1.0, 1.0), (-0.5, 0.5), (-1.0, 1.0)]

    # Build target waveform from the same simulator (noise-free)
    target = get_response(lam, true_volts)
    curve_obj = CurveObjective(lambda_ref=lam, target_ref=target)

    # Hardware (noise-free, bounded)
    hw = MockHardware(dac_size=3, wavelength=lam, noise_std=None, rng=rng, voltage_bounds=bounds)
    hw_obj = HardwareObjective(hw, curve_obj)

    # Initial point away from optimum
    x0 = np.array([0.0, 0.0, 0.0])
    y0, _ = hw_obj(x0)

    # Run Bayesian optimisation with skopt using hardware bounds
    bo = BayesianOptimizer(hw_obj, dimensions=bounds, base_estimator="GP", acq_func="EI", random_state=42)
    result = bo.run(n_calls=15, x0=x0)

    # Check loss reduced significantly and points stayed within bounds
    best_loss = result["best_loss"]
    assert best_loss < y0 * 0.5, f"Expected loss to decrease by >50%, got {best_loss} vs {y0}"

    history = result["history"]
    assert len(history) >= 1
    for h in history:
        x = np.asarray(h["x"], dtype=float)
        for xi, (lo, hi) in zip(x, bounds):
            assert lo - 1e-12 <= xi <= hi + 1e-12

    # Optional: check the best x is reasonably close to true (not too strict)
    best_x = result["best_x"]
    assert np.linalg.norm(best_x - true_volts) < 1.0  # loose sanity bound
