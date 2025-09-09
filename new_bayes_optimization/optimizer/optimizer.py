import numpy as np
from typing import Optional, Sequence, Tuple, Any
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
