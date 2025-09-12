import time
import numpy as np
import pytest


def _make_app():
    fastapi = pytest.importorskip("fastapi")
    from OCCS.service.app import create_app
    return create_app()


def test_rest_session_lifecycle_and_optimize_flow():
    from fastapi.testclient import TestClient

    app = _make_app()
    client = TestClient(app)

    # Backends list
    r = client.get("/api/backends")
    assert r.status_code == 200
    names = {b["name"] for b in r.json()}
    assert "mock" in names

    # Create session
    payload = {
        "backend": "mock",
        "dac_size": 3,
        "wavelength": {"start": 1.55e-6, "stop": 1.56e-6, "M": 96},
        "bounds": (-1.0, 1.0),
        "optimizer": {"random_state": 0, "acq_func": "gp_hedge"},
    }
    r = client.post("/api/session", json=payload)
    assert r.status_code == 200
    sid = r.json()["session_id"]

    # Initial status
    r = client.get(f"/api/session/{sid}/status")
    st = r.json()
    assert st["running"] is False
    assert st["iter"] == 0

    # Apply manual voltages
    r = client.post(f"/api/session/{sid}/voltages", json={"volts": [0.0, 0.0, 0.0]})
    assert r.status_code == 200 and r.json()["ok"] is True

    # Read waveform
    r = client.get(f"/api/session/{sid}/response")
    wf = r.json()
    assert len(wf["lambda"]) == 96
    assert len(wf["signal"]) == 96

    # Start a short optimization
    r = client.post(f"/api/session/{sid}/optimize/start", json={"n_calls": 3, "random_state": 1})
    assert r.status_code == 200 and r.json()["ok"] is True

    # Wait until done (or timeout)
    for _ in range(200):  # up to ~2s
        st = client.get(f"/api/session/{sid}/status").json()
        if not st.get("running", False) and st.get("iter", 0) >= 3:
            break
        time.sleep(0.01)
    else:
        pytest.fail("optimization did not finish in time")

    # History endpoint
    r = client.get(f"/api/session/{sid}/history")
    hist = r.json()
    assert isinstance(hist.get("history"), list)
    assert len(hist["history"]) >= 3
    assert hist["best_loss"] is None or isinstance(hist["best_loss"], float)

    # Stop (no-op if already finished)
    r = client.post(f"/api/session/{sid}/optimize/stop")
    assert r.status_code == 200 and r.json()["ok"] is True

    # Delete session
    r = client.delete(f"/api/session/{sid}")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_ws_status_connect_only():
    from fastapi.testclient import TestClient

    app = _make_app()
    client = TestClient(app)

    # Create session minimal
    payload = {
        "backend": "mock",
        "dac_size": 2,
        "wavelength": {"start": 1.55e-6, "stop": 1.551e-6, "M": 32},
    }
    sid = client.post("/api/session", json=payload).json()["session_id"]

    # Connect WS and receive initial status
    with client.websocket_connect(f"/api/session/{sid}/stream") as ws:
        msg = ws.receive_json()
        assert msg.get("type") == "status"
        assert "running" in msg

