"""Web service layer for OCCS (FastAPI app, session and hardware factory).

This package is designed to be optional: importing its modules should not
force optional dependencies like FastAPI or uvicorn at import time. The
`app.create_app()` factory performs lazy imports when you actually run the
web service.
"""

__all__ = [
    "app",
    "hardware",
    "models",
    "session",
]

