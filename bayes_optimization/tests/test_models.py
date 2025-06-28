import numpy as np
from bayes_optimization.bayes_optimizer.models import GaussianProcess

def test_gp_fit_predict():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(20, 1))
    y = np.sin(2 * np.pi * X).ravel()
    gp = GaussianProcess()
    gp.fit(X, y)
    pred, _ = gp.predict(X, return_std=True)
    mse = np.mean((pred - y) ** 2)
    assert mse < 1e-3
