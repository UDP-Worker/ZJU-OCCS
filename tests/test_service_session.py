import numpy as np

from OCCS.service.session import OptimizerSession


def test_session_create_and_waveform_with_mock():
    lam = np.linspace(1.55e-6, 1.56e-6, 200)
    sess = OptimizerSession(
        backend="mock",
        dac_size=3,
        wavelength=lam,
        bounds=[(-1.0, 1.0)] * 3,
        # Use default built-in target CSV in session when None
        target_csv_path=None,
        optimizer_kwargs={"random_state": 42, "acq_func": "gp_hedge"},
    )

    # Apply manual voltages and read back waveform
    sess.apply_manual([0.0, 0.0, 0.0])
    wf = sess.read_waveform()
    assert set(wf.keys()) >= {"lambda", "signal", "target"}
    assert isinstance(wf["lambda"], np.ndarray)
    assert isinstance(wf["signal"], np.ndarray)
    assert wf["lambda"].shape == wf["signal"].shape

    st = sess.status()
    assert st["running"] is False
    assert st["iter"] == 0

