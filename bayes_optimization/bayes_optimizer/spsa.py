"""Simplified SPSA refinement implementation."""

from typing import Callable
import numpy as np


def spsa_refine(
    start: np.ndarray, loss_fn: Callable[[np.ndarray], float], a0: float, c0: float, steps: int
) -> np.ndarray:
    """Perform SPSA minimization starting from ``start``."""

    x = np.asarray(start, dtype=float)
    dim = len(x)
    for k in range(steps):
        ak = a0 / (k + 1)
        ck = c0 / np.sqrt(k + 1)
        delta = np.random.choice([-1.0, 1.0], size=dim)
        loss_plus = loss_fn(x + ck * delta)
        loss_minus = loss_fn(x - ck * delta)
        grad = (loss_plus - loss_minus) / (2 * ck) * delta
        x = x - ak * grad
    return x
