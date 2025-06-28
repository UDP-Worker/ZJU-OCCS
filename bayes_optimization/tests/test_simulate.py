import numpy as np
from bayes_optimization.bayes_optimizer.simulate.optical_chip import (
    response,
    _IDEAL_RESPONSE,
)
from bayes_optimization.bayes_optimizer import config


def test_simulate_response_shape():
    w, resp = response(np.zeros(config.NUM_CHANNELS))
    assert resp.shape == _IDEAL_RESPONSE.shape
    assert w.shape == _IDEAL_RESPONSE.shape
    assert np.all(np.isfinite(resp))
