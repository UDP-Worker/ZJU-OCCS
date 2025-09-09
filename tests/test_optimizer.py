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
    """Use only HardwareObjective loss: ensure strong reduction and valid bounds."""
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
    hw = MockHardware(
        dac_size=3, wavelength=lam, noise_std=None, rng=rng, voltage_bounds=bounds
    )
    hw_obj = HardwareObjective(hw, curve_obj)

    # Initial point away from optimum
    x0 = np.array([0.0, 0.0, 0.0])
    y0, _ = hw_obj(x0)

    # Run Bayesian optimisation with skopt using hardware bounds
    bo = BayesianOptimizer(
        hw_obj, dimensions=bounds, base_estimator="GP", acq_func="EI", random_state=42
    )
    result = bo.run(n_calls=15, x0=x0)

    # Check points stayed within bounds
    history = result["history"]
    assert len(history) >= 1
    for h in history:
        x = np.asarray(h["x"], dtype=float)
        for xi, (lo, hi) in zip(x, bounds):
            assert lo - 1e-12 <= xi <= hi + 1e-12

    # Effectiveness: require a strong reduction relative to the initial loss
    best_loss = result["best_loss"]
    assert best_loss <= y0 * 0.1, f"Loss not reduced enough: {best_loss} vs initial {y0}"


def test_moderate_channels_show_improvement():
    """With 8 channels, the optimizer should still improve loss noticeably.

    Keep runtime modest: use a limited number of calls and a loose threshold.
    """
    rng = np.random.default_rng(0)
    lam = np.linspace(1.55e-6, 1.56e-6, 200)
    dac_size = 8
    # Fixed true vector for determinism
    true_volts = rng.uniform(-0.8, 0.8, size=dac_size)
    bounds = [(-1.0, 1.0)] * dac_size

    target = get_response(lam, true_volts)
    curve_obj = CurveObjective(lambda_ref=lam, target_ref=target)
    hw = MockHardware(dac_size=dac_size, wavelength=lam, noise_std=None, rng=rng, voltage_bounds=bounds)
    hw_obj = HardwareObjective(hw, curve_obj)

    x0 = np.zeros(dac_size)
    y0, _ = hw_obj(x0)

    bo = BayesianOptimizer(hw_obj, dimensions=bounds, base_estimator="GP", acq_func="EI", random_state=42)
    result = bo.run(n_calls=40, x0=x0)
    best_loss = result["best_loss"]

    # Expect at least a modest improvement without being too strict
    assert best_loss < y0 * 0.9, f"Expected some improvement, got {best_loss} vs {y0}"


def test_more_iterations_help_on_8_channels():
    """With 8 channels, more BO calls should yield lower loss (objective-based)."""
    rng = np.random.default_rng(1)
    lam = np.linspace(1.55e-6, 1.56e-6, 200)
    dac_size = 8
    true_volts = rng.uniform(-0.8, 0.8, size=dac_size)
    bounds = [(-1.0, 1.0)] * dac_size

    target = get_response(lam, true_volts)
    curve_obj = CurveObjective(lambda_ref=lam, target_ref=target)
    hw = MockHardware(dac_size=dac_size, wavelength=lam, noise_std=None, rng=rng, voltage_bounds=bounds)
    hw_obj = HardwareObjective(hw, curve_obj)

    x0 = np.zeros(dac_size)
    bo = BayesianOptimizer(hw_obj, dimensions=bounds, base_estimator="GP", acq_func="EI", random_state=42)
    res10 = bo.run(n_calls=10, x0=x0)
    # fresh optimizer to avoid carry-over state
    bo2 = BayesianOptimizer(hw_obj, dimensions=bounds, base_estimator="GP", acq_func="EI", random_state=42)
    res40 = bo2.run(n_calls=40, x0=x0)

    assert res40["best_loss"] <= res10["best_loss"] * 0.9, (
        f"More calls should help: 40-calls best={res40['best_loss']} vs 10-calls best={res10['best_loss']}"
    )
