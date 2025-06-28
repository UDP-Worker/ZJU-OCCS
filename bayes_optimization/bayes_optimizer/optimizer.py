"""Simplified Bayesian optimization loop."""

from typing import Callable, Any, Dict
import numpy as np

class BayesOptimizer:
    def __init__(self, gp, acq_fn: Callable, bounds: np.ndarray, *, candidate_factor: int = 4):
        self.gp = gp
        self.acq_fn = acq_fn
        self.bounds = np.asarray(bounds, dtype=float)
        self.candidate_factor = int(candidate_factor)

    def optimize(
        self, start: np.ndarray, loss_fn: Callable[[np.ndarray], float], steps: int
    ) -> Dict[str, Any]:
        """Run a very basic BO loop using random search."""

        dim = len(start)
        X = [np.asarray(start, dtype=float)]
        y = [loss_fn(start)]
        self.gp.fit(np.vstack(X), np.array(y))
        best_x, best_y = X[0], y[0]

        base = max(64, self.candidate_factor * dim)
        for _ in range(steps - 1):
            candidates = np.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1], size=(base, dim)
            )
            mu, sigma = self.gp.predict(candidates, return_std=True)
            scores = self.acq_fn(mu, sigma, best_y)
            x_next = candidates[np.argmax(scores)]
            y_next = loss_fn(x_next)
            X.append(x_next)
            y.append(y_next)
            self.gp.fit(np.vstack(X), np.array(y))
            if y_next < best_y:
                best_x, best_y = x_next, y_next

        return {"best_x": best_x, "best_y": best_y}
