"""Simple Gaussian process wrapper using scikit-learn."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF


class GaussianProcess:
    """Thin wrapper around :class:`GaussianProcessRegressor`."""

    def __init__(self, kernel: str = "Matern52"):
        if kernel == "Matern52":
            kern = Matern(nu=2.5)
        elif kernel == "RBF":
            kern = RBF()
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        self.gp = GaussianProcessRegressor(kernel=kern, normalize_y=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP to training data."""
        self.gp.fit(X, y)

    def predict(self, X: np.ndarray, return_std: bool = True):
        """Predict mean and optionally standard deviation."""
        return self.gp.predict(X, return_std=return_std)
