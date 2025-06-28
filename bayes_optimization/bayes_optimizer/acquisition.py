"""Common acquisition functions."""

import numpy as np


from scipy.stats import norm


def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01
) -> np.ndarray:
    """Compute the expected improvement."""
    sigma = np.maximum(sigma, 1e-12)
    improvement = best - mu - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


def trust_region_ei(mu: np.ndarray, sigma: np.ndarray, best: float, center: np.ndarray, candidates: np.ndarray, radius: float, xi: float = 0.01) -> np.ndarray:
    """EI restricted to a trust region around ``center``."""
    diffs = np.linalg.norm(candidates - center, axis=1)
    ei = expected_improvement(mu, sigma, best, xi)
    ei[diffs > radius] = -np.inf
    return ei
