"""FastAPI application factory for OCCS web service.

Phase 1 provides only a minimal app factory without binding full routes.
Imports FastAPI lazily to avoid hard dependency during tests.
"""

from __future__ import annotations

from typing import Any


def create_app() -> Any:
    try:
        from fastapi import FastAPI  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "FastAPI is required to run the web service. Install fastapi first."
        ) from exc

    app = FastAPI(title="OCCS Web Service", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    return app


# Convenience for manual launch: `python -m OCCS.service.app`
if __name__ == "__main__":  # pragma: no cover - manual run
    try:
        import uvicorn  # type: ignore
    except Exception:
        print("uvicorn is required to run the app: pip install uvicorn fastapi")
        raise SystemExit(1)
    uvicorn.run("OCCS.service.app:create_app", host="127.0.0.1", port=8000, factory=True)

