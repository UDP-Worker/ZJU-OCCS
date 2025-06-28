import numpy as np
from bayes_optimization.bayes_optimizer.simulate.optical_chip import (
    response,
    _IDEAL_RESPONSE,
    get_ideal_voltages,
)
from bayes_optimization.bayes_optimizer import config


def test_simulate_response_shape():
    w, resp = response(np.zeros(config.NUM_CHANNELS))
    assert resp.shape == _IDEAL_RESPONSE.shape
    assert w.shape == _IDEAL_RESPONSE.shape
    assert np.all(np.isfinite(resp))
    # zero voltages should not perfectly match the target waveform
    assert not np.allclose(resp, _IDEAL_RESPONSE)
    ideal = get_ideal_voltages(config.NUM_CHANNELS)
    _, resp2 = response(ideal)
    assert np.allclose(resp2, _IDEAL_RESPONSE)
