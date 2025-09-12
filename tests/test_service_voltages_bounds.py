import pytest


def _make_app():
    fastapi = pytest.importorskip("fastapi")
    from OCCS.service.app import create_app
    return create_app()


def test_get_voltages_and_bounds_clipping():
    from fastapi.testclient import TestClient

    app = _make_app()
    client = TestClient(app)

    payload = {
        "backend": "mock",
        "dac_size": 3,
        "wavelength": {"start": 1.55e-6, "stop": 1.56e-6, "M": 64},
        "bounds": [(-0.5, 0.5), (-1.0, 1.0), (0.0, 0.2)],
    }
    sid = client.post("/api/session", json=payload).json()["session_id"]

    # default voltages are zeros
    r = client.get(f"/api/session/{sid}/voltages")
    assert r.status_code == 200 and r.json().get("volts") == [0.0, 0.0, 0.0]

    # apply out-of-bounds and ensure clipping
    client.post(f"/api/session/{sid}/voltages", json={"volts": [1.0, -2.0, 0.5]})
    r = client.get(f"/api/session/{sid}/voltages")
    vv = r.json()["volts"]
    assert vv[0] == pytest.approx(0.5)
    assert vv[1] == pytest.approx(-1.0)
    assert vv[2] == pytest.approx(0.2)

