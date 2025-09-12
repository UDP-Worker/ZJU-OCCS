"""FastAPI application factory for OCCS web service.

Phase 2: add REST/WS endpoints for sessions, hardware operations, and
optimization control with basic validation. Imports FastAPI lazily to avoid
hard dependency for non-web usages.
"""

from __future__ import annotations

from typing import Any, Dict
import uuid
import asyncio
import numpy as np


def create_app() -> Any:
    try:
        # Import FastAPI and WebSocket lazily to keep dependency optional
        from fastapi import FastAPI, WebSocket  # type: ignore
        # Ensure the WebSocket type is available in module globals so that
        # FastAPI can resolve deferred annotations inside the factory.
        globals()["WebSocket"] = WebSocket
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "FastAPI is required to run the web service. Install fastapi first."
        ) from exc

    app = FastAPI(title="OCCS Web Service", version="0.2.0")

    # Enable permissive CORS for development/web UI usage.
    try:
        from fastapi.middleware.cors import CORSMiddleware  # type: ignore
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except Exception:
        # CORS is optional; if middleware import fails just continue.
        pass

    # Runtime session store: id -> OptimizerSession
    app.state.sessions = {}

    # ---- Utilities ----
    from OCCS.service.hardware import list_backends
    from OCCS.service.session import OptimizerSession
    from OCCS.service.models import WavelengthSpec, normalise_bounds

    def _ensure_session(sid: str) -> OptimizerSession:
        sess = app.state.sessions.get(sid)
        if sess is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="session not found")
        return sess  # type: ignore[return-value]

    def _to_list_array(obj):
        arr = np.asarray(obj)
        return arr.tolist()

    def _to_jsonable(obj):
        try:
            import numpy as _np  # local alias to avoid confusion
        except Exception:  # pragma: no cover
            _np = None  # type: ignore
        if _np is not None:
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        return obj

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    # ---- REST: Backends ----
    @app.get("/api/backends")
    def api_backends() -> list[dict]:
        return list_backends()

    # ---- REST: Session management ----
    @app.post("/api/session")
    async def api_create_session(payload: Dict[str, Any]) -> Dict[str, Any]:
        backend = str(payload.get("backend", "mock"))
        try:
            dac_size = int(payload["dac_size"])  # required
        except Exception:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="dac_size is required and must be int")

        wl = payload.get("wavelength")
        if wl is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="wavelength is required")
        if isinstance(wl, dict):
            spec = WavelengthSpec(array=None, start=wl.get("start"), stop=wl.get("stop"), M=wl.get("M"))
            wavelength = spec.resolve()
        else:
            wavelength = np.asarray(wl, dtype=float).ravel()
        if wavelength.size < 2:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="wavelength needs at least 2 points")

        bounds = normalise_bounds(payload.get("bounds"), dac_size) if payload.get("bounds") is not None else None
        target_csv_path = payload.get("target_csv_path")
        if target_csv_path is not None:
            try:
                from pathlib import Path
                p = Path(str(target_csv_path))
                if not (p.exists() and p.is_file()):
                    from fastapi import HTTPException
                    raise HTTPException(status_code=400, detail="target_csv_path not found or not a file")
            except Exception as _exc:
                # If path cannot be interpreted, surface as 400
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="invalid target_csv_path")
        opt = payload.get("optimizer") or {}
        optimizer_kwargs = {}
        if "acq_func" in opt:
            optimizer_kwargs["acq_func"] = opt.get("acq_func")
        if "random_state" in opt and opt.get("random_state") is not None:
            optimizer_kwargs["random_state"] = int(opt.get("random_state"))

        sess = OptimizerSession(
            backend=backend,
            dac_size=int(dac_size),
            wavelength=wavelength,
            bounds=bounds,
            target_csv_path=target_csv_path,
            optimizer_kwargs=optimizer_kwargs,
        )
        sid = uuid.uuid4().hex
        app.state.sessions[sid] = sess
        return {"session_id": sid}

    @app.get("/api/session/{sid}/status")
    def api_session_status(sid: str) -> Dict[str, Any]:
        sess = _ensure_session(sid)
        st = sess.status()
        st["running"] = bool(sess.running)
        if sess.best_loss is not None:
            st["best_loss"] = float(sess.best_loss)
        if sess.best_x is not None:
            st["x"] = list(map(float, np.asarray(sess.best_x)))
        return st

    @app.delete("/api/session/{sid}")
    def api_session_delete(sid: str) -> Dict[str, Any]:
        sess = _ensure_session(sid)
        sess.close()
        del app.state.sessions[sid]
        return {"ok": True}

    # ---- REST: Hardware operations ----
    @app.post("/api/session/{sid}/voltages")
    def api_apply_voltages(sid: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        sess = _ensure_session(sid)
        volts = payload.get("volts")
        if volts is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="volts is required")
        arr = np.asarray(volts, dtype=float).ravel()
        if arr.size != int(sess.dac_size):
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=f"expected {sess.dac_size} volt values")
        sess.apply_manual(arr)
        return {"ok": True}

    @app.get("/api/session/{sid}/response")
    def api_read_response(sid: str) -> Dict[str, Any]:
        sess = _ensure_session(sid)
        wf = sess.read_waveform()
        return {
            "lambda": _to_list_array(wf["lambda"]),
            "signal": _to_list_array(wf["signal"]),
            "target": _to_list_array(wf["target"]),
        }

    @app.get("/api/session/{sid}/voltages")
    def api_get_voltages(sid: str) -> Dict[str, Any]:
        sess = _ensure_session(sid)
        try:
            arr = sess.hardware.read_voltage()
        except Exception:
            import numpy as _np
            arr = _np.zeros(int(sess.dac_size), dtype=float)
        return {"volts": list(map(float, np.asarray(arr, dtype=float)))}

    # ---- REST: Optimization control ----
    @app.post("/api/session/{sid}/optimize/start")
    async def api_opt_start(sid: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        sess = _ensure_session(sid)
        n_calls = int(payload.get("n_calls", 0))
        if n_calls <= 0:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="n_calls must be positive")
        x0 = payload.get("x0")
        acq = payload.get("acq_func")
        rs = payload.get("random_state")
        try:
            sess.start_optimize(
                n_calls=n_calls,
                x0=x0,
                acq_func=acq,
                random_state=int(rs) if rs is not None else None,
            )
        except RuntimeError as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=409, detail=str(e))
        return {"ok": True}

    @app.post("/api/session/{sid}/optimize/stop")
    def api_opt_stop(sid: str) -> Dict[str, Any]:
        sess = _ensure_session(sid)
        sess.stop_optimize()
        return {"ok": True}

    @app.get("/api/session/{sid}/history")
    def api_history(sid: str) -> Dict[str, Any]:
        sess = _ensure_session(sid)
        hist = []
        for h in sess.history:
            hist.append({
                "x": list(map(float, np.asarray(h.get("x", [])))),
                "loss": float(h.get("loss", np.nan)),
                "diag": _to_jsonable(h.get("diag", {})),
            })
        return {
            "history": hist,
            "best_loss": float(sess.best_loss) if sess.best_loss is not None else None,
            "best_x": list(map(float, np.asarray(sess.best_x))) if sess.best_x is not None else None,
        }

    @app.get("/api/session/{sid}/history.csv")
    def api_history_csv(sid: str):
        """Return optimization history as CSV for download."""
        sess = _ensure_session(sid)
        import io, csv
        buf = io.StringIO()
        w = csv.writer(buf)
        # header
        cols = ["iter", "loss"] + [f"x{i}" for i in range(int(sess.dac_size))]
        w.writerow(cols)
        for i, h in enumerate(sess.history, start=1):
            x = list(map(float, np.asarray(h.get("x", []))))
            row = [i, float(h.get("loss", np.nan))] + x
            w.writerow(row)
        from fastapi import Response
        return Response(content=buf.getvalue(), media_type="text/csv")

    # ---- WebSocket: streaming ----
    @app.websocket("/api/session/{sid}/stream")
    async def api_stream(websocket: WebSocket, sid: str):
        await websocket.accept()
        # Look up session after accepting to avoid handshake issues
        try:
            sess = _ensure_session(sid)
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            try:
                await websocket.close()
            except Exception:
                pass
            return
        # Emit immediate status snapshot
        st = sess.status()
        st["running"] = bool(sess.running)
        if sess.best_x is not None:
            st["x"] = list(map(float, np.asarray(sess.best_x)))
        await websocket.send_json({"type": "status", **st})

        # Attach subscriber queue for future events
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        sess.add_subscriber(loop, q)
        try:
            while True:
                event = await q.get()
                await websocket.send_json(event)
        except Exception:
            # On disconnect, detach
            sess.remove_subscriber(q)
            try:
                await websocket.close()
            except Exception:
                pass
            return

    # ---- Static: mount built front-end if available ----
    try:
        from starlette.staticfiles import StaticFiles  # type: ignore
        import os
        from pathlib import Path
        webui_dist = Path(__file__).resolve().parent.parent / "webui" / "dist"
        if webui_dist.exists() and webui_dist.is_dir():
            app.mount("/", StaticFiles(directory=str(webui_dist), html=True), name="webui")
    except Exception:
        # Static hosting is optional for development
        pass

    return app


# Convenience for manual launch: `python -m OCCS.service.app`
if __name__ == "__main__":  # pragma: no cover - manual run
    try:
        import uvicorn  # type: ignore
    except Exception:
        print("uvicorn is required to run the app: pip install uvicorn fastapi")
        raise SystemExit(1)
    uvicorn.run("OCCS.service.app:create_app", host="127.0.0.1", port=8000, factory=True)
