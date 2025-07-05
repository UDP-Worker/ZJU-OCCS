import numpy as np
from bayes_optimization.bayes_optimizer.simulate import optical_chip
from bayes_optimization.bayes_optimizer import config


def test_simulate_response_shape():
    target_wl, target_resp = optical_chip.get_target_waveform()
    w, resp = optical_chip.response(np.zeros(config.NUM_CHANNELS), target_wl)
    assert resp.shape == target_resp.shape
    assert w.shape == target_wl.shape
    assert np.all(np.isfinite(resp))
    # zero voltages should not perfectly match the target waveform
    assert not np.allclose(resp, target_resp)
    ideal = optical_chip.get_ideal_voltages(config.NUM_CHANNELS)
    _, resp2 = optical_chip.response(ideal, target_wl)
    assert resp2.shape == target_resp.shape
