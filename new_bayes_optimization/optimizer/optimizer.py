import numpy as np
from typing import Optional, Sequence, Tuple, Any, Dict, List
from skopt import Optimizer

from new_bayes_optimization.optimizer.objective import HardwareObjective


class BayesianOptimizer:

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
            record(x0, loss0, diag0)

        for _ in range(int(n_calls)):
            x = self.suggest()
            loss, diag = self.hardware_objective(x)
            self.observe(x, loss)
            record(x, loss, diag)

        # Determine the best from our history
        best_idx = int(np.argmin([h["loss"] for h in history])) if history else -1
        best = history[best_idx] if history else {"x": None, "loss": np.inf}
        return {
            "best_x": best["x"],
            "best_loss": best["loss"],
            "history": history,
        }
