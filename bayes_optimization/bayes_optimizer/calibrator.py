"""Calibration utilities (placeholder)."""

import numpy as np


def measure_jacobian(n_samples: int | None = None) -> np.ndarray:
    """Return a mock sensitivity matrix."""
    raise NotImplementedError


def compress_modes(J: np.ndarray, var_ratio: float = 0.95):
    """Placeholder for PCA compression."""
    raise NotImplementedError
