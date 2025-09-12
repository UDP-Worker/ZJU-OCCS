import sys
from io import StringIO

from OCCS.cli import web_main


def test_cli_missing_uvicorn_returns_error_code(monkeypatch):
    # Simulate ImportError for uvicorn by deleting from sys.modules and masking import
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # pragma: no cover - simple shim
        if name == "uvicorn":
            raise ImportError("No module named 'uvicorn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    stderr = StringIO()
    real_stderr = sys.stderr
    try:
        sys.stderr = stderr
        code = web_main(["--host", "127.0.0.1", "--port", "9000"])  # should not try to run
    finally:
        sys.stderr = real_stderr

    assert code != 0
    assert "uvicorn" in stderr.getvalue()

