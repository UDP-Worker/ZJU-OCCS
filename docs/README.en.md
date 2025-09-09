# OCCS: Optical Chip Curve Search

[English](./README.en.md) | [简体中文](./README.zh-CN.md)

Lightweight simulator + Bayesian optimization toolkit for tuning optical chip voltages so that the resulting spectrum matches a target curve. The codebase is modular and test-backed, with utilities to log and visualize optimization progress, including GP uncertainty.

## Key Features

- Mock hardware and simple, deterministic optical response simulator
- Curve-similarity objective with alignment, robust loss, and optional band weights
- skopt-based Bayesian optimization with per-iteration GP max uncertainty logging
- Plots (loss, running min, GP uncertainty) and CSV export for reproducible runs

## Project Structure

- `OCCS/`
  - `connector/`: Hardware adapters
    - `mock_hardware.py`: In-memory mock device used in tests and examples
    - `real_hardware.py`: API placeholder for a real DAC/OSA stack
  - `simulate/`: Simple optical response model (`get_response`)
  - `optimizer/`: Objective, optimizer wrapper, and visualization helpers
    - `objective.py`: CurveObjective/HardwareObjective and CSV-based builders
    - `optimizer.py`: Thin wrapper over `skopt.Optimizer` with logging of GP uncertainty
    - `viz.py`: Plot loss and GP uncertainty, export CSV logs
  - `data/optimization/`: Default output location for plots/CSV created by tests or examples
- `tests/`: Pytest-based tests that also generate example outputs (plots + logs)
- `environment.yml`: Conda environment file (NumPy, scikit-optimize, Matplotlib, PyTest, etc.)

## Quick Start

1) Create environment

```bash
# Using conda/mamba
mamba env create -f environment.yml  # or: conda env create -f environment.yml
mamba activate ZJU-OCCS             # or: conda activate ZJU-OCCS
```

2) Run tests (also generates example outputs)

```bash
pytest -q
```

Outputs appear under `OCCS/data/optimization/`:

- `loss_curve_*.png`: Loss vs iteration with running min; GP max std overlay when available
- `log_*.csv`: Iteration-by-iteration logs including voltages, loss, delta_nm, and GP uncertainty (`gp_max_std`, `gp_max_var`)

3) Minimal example (script excerpt)

```python
import numpy as np
from OCCS.connector import MockHardware
from OCCS.optimizer.objective import create_hardware_objective
from OCCS.optimizer.optimizer import BayesianOptimizer
from OCCS.optimizer.viz import save_loss_history_plot, save_uncertainty_history_plot, save_history_csv

lam = np.linspace(1.55e-6, 1.56e-6, 200)
bounds = [(-1.0, 1.0)] * 3
hw = MockHardware(dac_size=3, wavelength=lam, voltage_bounds=bounds)

# Build objective from a two-row CSV (first row: wavelength, second: target)
obj = create_hardware_objective(hw, target_csv_path="OCCS/data/ideal_waveform.csv", M=200)

bo = BayesianOptimizer(obj, dimensions=bounds, base_estimator="GP", acq_func="EI", random_state=42)
result = bo.run(n_calls=30, x0=[0.0, 0.0, 0.0])

save_loss_history_plot(result, "OCCS/data/optimization/loss_curve_example.png", title="BO Loss")
save_uncertainty_history_plot(result, "OCCS/data/optimization/gp_uncertainty.png", metric="std", title="GP Max Std")
save_history_csv(result, "OCCS/data/optimization/log_example.csv")
```

## Implementation Notes

- The GP max uncertainty per iteration is estimated via random sampling over the bounded space using the current surrogate; early iterations may show NaN until the GP is fitted.
- The objective operates on normalised shapes with optional small-range wavelength alignment and Huber loss; see `ObjectiveConfig` for tuning.

## Contributing

- Code style: NumPy-style docstrings; prefer small, focused modules and tests.
- Tests: run `pytest -q`. Contributions that add features should include tests and minimal docs.

## License

This project is licensed under the terms of the LICENSE file in this repository.

