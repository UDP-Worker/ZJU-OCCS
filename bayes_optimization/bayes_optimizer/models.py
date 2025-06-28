"""Gaussian process models (placeholder)."""

import numpy as np


class GaussianProcess:
    def __init__(self, kernel: str = "Matern52"):
        self.kernel = kernel

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray, return_std: bool = True):
        raise NotImplementedError
