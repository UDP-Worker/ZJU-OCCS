import sys
from io import StringIO


def test_create_app_import_guard(monkeypatch):
    """create_app should raise ImportError with a helpful message if FastAPI is missing."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # pragma: no cover - shim
        if name == "fastapi":
            raise ImportError("No module named 'fastapi'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Importing the module should still work (lazy import inside create_app)
    from OCCS.service.app import create_app

    try:
        create_app()
    except ImportError as e:
        assert "FastAPI is required" in str(e)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ImportError when FastAPI is unavailable")


def test_health_endpoint_when_fastapi_available():
    """If FastAPI is installed, /api/health should respond with ok=True."""
    import pytest

    fastapi = pytest.importorskip("fastapi")  # skip test gracefully if not installed
    from fastapi.testclient import TestClient
    from OCCS.service.app import create_app

    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is True

