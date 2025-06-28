"""Bayesian optimization loop (placeholder)."""

from typing import Callable, Any, Dict
import numpy as np

class BayesOptimizer:
    def __init__(self, gp, acq_fn: Callable, bounds: np.ndarray):
        self.gp = gp
        self.acq_fn = acq_fn
        self.bounds = bounds

    def optimize(self, start: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError
