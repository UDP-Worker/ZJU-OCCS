from __future__ import annotations

import sys
from typing import Optional


def web_main(argv: Optional[list[str]] = None) -> int:
    """Run the OCCS web service.

    Returns process exit code instead of calling sys.exit directly to ease testing.
    """
    argv = argv if argv is not None else sys.argv[1:]
    host = "127.0.0.1"
    port = 8000
    # Minimal arg parsing: --host X --port Y
    it = iter(argv)
    for a in it:
        if a == "--host":
            try:
                host = next(it)
            except StopIteration:
                print("Missing value for --host", file=sys.stderr)
                return 2
        elif a == "--port":
            try:
                port = int(next(it))
            except StopIteration:
                print("Missing value for --port", file=sys.stderr)
                return 2
            except Exception:
                print("Invalid port value", file=sys.stderr)
                return 2

    try:
        import uvicorn  # type: ignore
    except Exception:
        print("uvicorn is required. Please install fastapi and uvicorn.", file=sys.stderr)
        return 1
    try:
        from OCCS.service.app import create_app
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    uvicorn.run("OCCS.service.app:create_app", host=host, port=port, factory=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(web_main())

