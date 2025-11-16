# OCCS: Optical Chip Control & Search (Web UI + FastAPI)

[English](./README.en.md) | [简体中文](./README.zh-CN.md)

OCCS provides an interactive web application and a Python service layer to control optical hardware (mock or real) and optimize voltages so the measured spectrum matches a target curve. The web UI streams waveforms and optimization diagnostics in real time, while the core library remains reusable and test-backed.

## Highlights

- Web UI (Vite + React + TypeScript) for live waveform and loss charts
- FastAPI backend with REST + WebSocket for sessions and real-time progress
- Manual voltage control and bounds enforcement; CSV history download
- Core modules for simulation, objectives, and skopt-based Bayesian optimization

## Architecture

- `OCCS/optimizer`: objective functions, optimizer wrapper, and offline visualization
- `OCCS/connector`: hardware interfaces (`MockHardware` and `RealHardware` placeholder)
- `OCCS/service`: FastAPI app, session manager, and hardware factory
  - REST: create session, set voltages, fetch waveform, start/stop optimization, export history
  - WebSocket: stream progress, waveform snapshots, and status
- `OCCS/webui`: single-page app served by the backend when built

## Quick Start (Docker, recommended)

Use prebuilt image from GitHub Releases only:

```bash
# 1) Download occs-web.tar.gz from Release assets
# 2) Load the image
docker load -i occs-web.tar.gz

# docker will print the image name:tag (e.g., occs-web:0.1.1)
# 3) Run it (replace <image:tag> accordingly)
docker run --rm -p 8000:8000 <image:tag>

# Optional: enable real hardware and persist uploads
docker run --rm -e OCCS_REAL_AVAILABLE=1 \
  -v $(pwd)/uploads:/tmp/occs_uploads \
  -p 8000:8000 <image:tag>
```

### Using the app

- Select backend (mock by default). Real hardware is disabled unless `OCCS_REAL_AVAILABLE=1`.
- Configure wavelength grid and voltage bounds; optionally upload a target CSV.
- Create a session, adjust voltages manually, and refresh waveforms.
- Start optimization to stream loss and diagnostics; download history CSV anytime.

## Front-end development on another machine

1. Backend host: set up the Python environment (Conda, venv, or uv as described in the repo README) and launch the FastAPI service with `occs-web --host 0.0.0.0 --port 8000`. Keep port 8000 open so the remote UI can reach it.
2. Front-end host: clone this repository (or copy the `OCCS/webui` folder) and install Node.js 20+. Run `cd OCCS/webui && npm install` to fetch dependencies.
3. Edit `OCCS/webui/vite.config.ts` so the dev proxy `target` points to the backend machine instead of `127.0.0.1` (e.g., `http://192.168.1.50:8000`). This ensures REST + WebSocket traffic is forwarded correctly during development.
4. Start the Vite server with `npm run dev -- --host 0.0.0.0 --port 5173`. Open `http://<frontend-host>:5173/` from your browser; the UI will talk to the remote backend through the proxy.
5. When you need the backend to serve the updated UI, run `npm run build` and copy the generated `OCCS/webui/dist` directory back to the backend host (or keep using the dev server for live reloads).

## API Overview (REST)

- `GET /api/backends` → available backends
- `POST /api/session` → create session; returns `{ session_id }`
- `GET /api/session/{id}/status` → running state, iter, best loss, current best x
- `DELETE /api/session/{id}` → close the session
- `POST /api/session/{id}/voltages` → set voltages
- `GET /api/session/{id}/response` → waveform `{ lambda, signal, target }`
- `POST /api/session/{id}/optimize/start` → start optimization
- `POST /api/session/{id}/optimize/stop` → stop optimization
- `GET /api/session/{id}/history(.csv)` → fetch/export history

WebSocket: `GET /api/session/{id}/stream` → emits `status`, `progress`, `waveform`, `done`, `error`.

## Development & Tests

- Run tests (also generates sample plots/CSVs): `pytest -q`
- Linting/formatting: follow existing code style; add tests for new features

## License

This project is licensed under the terms of the LICENSE file in this repository.
