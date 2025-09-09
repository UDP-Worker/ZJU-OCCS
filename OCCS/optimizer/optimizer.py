"""Bayesian optimization wrapper around scikit-optimize for OCCS hardware."""

import logging
from typing import Optional, Sequence, Tuple, Any, Dict, List

import numpy as np
from skopt import Optimizer

from OCCS.optimizer.objective import HardwareObjective

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """Manage suggest-observe loop and basic diagnostics.

    Wraps a ``skopt.Optimizer`` configured with voltage bounds and offers
    convenience helpers to run multiple iterations while tracking simple
    diagnostics like an estimate of the model's maximum uncertainty.
    """

    def __init__(
        self,
        hardware_objective: HardwareObjective,
        dimensions: Optional[Sequence[Tuple[float, float]]] = None,
        **skopt_kwargs: Any,
    ) -> None:
        """
        Thin wrapper holding a skopt.Optimizer configured with hardware bounds.

        Parameters
        ----------
        hardware_objective:
            Coupled objective providing access to hardware and loss function.
        dimensions:
            skopt-compatible bounds. If None, tries to read from
            ``hardware_objective.hardware.voltage_bounds``.
        skopt_kwargs:
            Additional kwargs forwarded to ``skopt.Optimizer``.
        """
        self.hardware_objective = hardware_objective
        if dimensions is None:
            dimensions = getattr(hardware_objective.hardware, "voltage_bounds", None)
        self.dimensions = dimensions
        self._opt: Optional[Optimizer] = None
        if self.dimensions is not None:
            self._opt = Optimizer(dimensions=self.dimensions, **skopt_kwargs)

    @property
    def optimizer(self) -> Optimizer:
        """Return the underlying ``skopt.Optimizer`` instance."""
        if self._opt is None:
            raise RuntimeError("Optimizer not initialised: missing dimensions/bounds.")
        return self._opt

    def suggest(self) -> np.ndarray:
        """Ask skopt for the next candidate voltage vector."""
        x = self.optimizer.ask()
        return np.asarray(x, dtype=float)

    def observe(self, x: Sequence[float], y: float) -> None:
        """Report observation back to skopt."""
        self.optimizer.tell(list(map(float, x)), float(y))

    def step(self, x: Optional[Sequence[float]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the objective at ``x`` (or at a suggested point if None),
        report it to skopt and return (loss, diag).
        """
        if x is None:
            x = self.suggest()
        x = np.asarray(x, dtype=float)
        loss, diag = self.hardware_objective(x)
        self.observe(x, loss)
        return loss, diag

    def _sample_points(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` random points uniformly from the bounded search space.

        Returns an array of shape (n, d). If dimensions are not available,
        returns an empty array.
        """
        if not self.dimensions:
            return np.empty((0, 0))
        dims = list(self.dimensions)
        d = len(dims)
        x_mat = np.empty((n, d), dtype=float)
        for j, (low, high) in enumerate(dims):
            x_mat[:, j] = rng.uniform(low, high, size=n)
        return x_mat

    def _compute_gp_max_uncertainty(
        self, n_samples: int = 1024, seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """Estimate the GP's maximum predictive uncertainty over the space.

        Uses random sampling over the bounded space to approximate the maximum
        posterior standard deviation and variance of the current surrogate
        model. Returns (max_std, max_var). If no model is available, returns
        (nan, nan).
        """
        try:
            from typing import cast

            models_obj = getattr(self.optimizer, "models", None)
            if not models_obj:
                return float("nan"), float("nan")
            models_list = cast(list, models_obj)
            model = models_list[-1]
        except Exception:
            return float("nan"), float("nan")

        # If there is no bounded space, cannot sample meaningfully
        if not self.dimensions:
            return float("nan"), float("nan")

        rng = np.random.default_rng(seed)
        x_samples = self._sample_points(int(n_samples), rng)
        if x_samples.size == 0:
            return float("nan"), float("nan")

        # GaussianProcessRegressor supports return_std=True
        try:
            _, std = model.predict(x_samples, return_std=True)
        except Exception:
            return float("nan"), float("nan")
        max_std = float(np.max(std)) if std.size else float("nan")
        return max_std, float(max_std ** 2)

    def run(
        self,
        n_calls: int,
        x0: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        Run ``n_calls`` evaluations. If ``x0`` is provided, evaluate it first.

        Returns a dict with keys: best_x, best_loss, history (list of steps).
        Each history item contains keys: x, loss, diag.
        """
        history: List[Dict[str, Any]] = []

        def record(x: np.ndarray, loss: float, diag: Dict[str, Any]):
            history.append({"x": np.asarray(x, dtype=float), "loss": float(loss), "diag": diag})

        if x0 is not None:
            x0 = np.asarray(x0, dtype=float)
            loss0, diag0 = self.hardware_objective(x0)
            # Ensure internal model sees this observation first
            self.observe(x0, loss0)
            # Compute initial GP uncertainty after seeding with x0
            max_std0, max_var0 = self._compute_gp_max_uncertainty(n_samples=1024, seed=12345)
            diag0["gp_max_std"] = max_std0
            diag0["gp_max_var"] = max_var0
            try:
                logger.info(
                    "Init | loss=%.6g | GP max std=%.6g (var=%.6g)",
                    float(loss0), max_std0, max_var0,
                )
            except Exception:
                print(
                    f"Init | loss={float(loss0):.6g} | GP max std={max_std0:.6g} "
                    f"(var={max_var0:.6g})"
                )
            record(x0, loss0, diag0)

        for it in range(1, int(n_calls) + 1):
            x = self.suggest()
            loss, diag = self.hardware_objective(x)
            self.observe(x, loss)
            # After updating the surrogate, estimate its max uncertainty
            # Use a deterministic seed per-iteration for reproducibility
            max_std, max_var = self._compute_gp_max_uncertainty(
                n_samples=1024, seed=12345 + it
            )
            diag["gp_max_std"] = max_std
            diag["gp_max_var"] = max_var
            try:
                logger.info(
                    "Iter %d | loss=%.6g | GP max std=%.6g (var=%.6g)",
                    it, float(loss), max_std, max_var,
                )
            except Exception:
                # Fallback printing if logging not configured
                print(
                    f"Iter {it} | loss={float(loss):.6g} | GP max std={max_std:.6g} "
                    f"(var={max_var:.6g})"
                )
            record(x, loss, diag)

        # Determine the best from our history
        best_idx = int(np.argmin([h["loss"] for h in history])) if history else -1
        best = history[best_idx] if history else {"x": None, "loss": np.inf}
        return {
            "best_x": best["x"],
            "best_loss": best["loss"],
            "history": history,
        }
