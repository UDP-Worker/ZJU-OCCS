"""Acquisition functions (placeholder)."""

import numpy as np


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    raise NotImplementedError


def trust_region_ei(*args, **kwargs):
    raise NotImplementedError
