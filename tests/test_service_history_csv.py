import time
import pytest


def _make_app():
    fastapi = pytest.importorskip("fastapi")
    from OCCS.service.app import create_app
    return create_app()


def test_history_csv_download_after_optimize():
    from fastapi.testclient import TestClient

    app = _make_app()
    client = TestClient(app)

    # Create session
    payload = {
        "backend": "mock",
        "dac_size": 2,
        "wavelength": {"start": 1.55e-6, "stop": 1.551e-6, "M": 48},
    }
    sid = client.post("/api/session", json=payload).json()["session_id"]

    # Start short optimization
    r = client.post(f"/api/session/{sid}/optimize/start", json={"n_calls": 2, "random_state": 0})
    assert r.status_code == 200

    # Wait until done
    for _ in range(200):
        st = client.get(f"/api/session/{sid}/status").json()
        if not st.get("running", False) and st.get("iter", 0) >= 2:
            break
        time.sleep(0.01)
    else:
        pytest.fail("optimization did not finish in time")

    # Download CSV
    r = client.get(f"/api/session/{sid}/history.csv")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/csv")
    text = r.text.strip().splitlines()
    assert len(text) >= 2  # header + at least one row

