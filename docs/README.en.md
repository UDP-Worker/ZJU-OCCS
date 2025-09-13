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

## Quick Start

### 1) Python dependencies

Use Conda/Mamba (recommended):

```bash
mamba env create -f environment.yml  # or: conda env create -f environment.yml
mamba activate ZJU-OCCS

# For the web service (option A)
pip install "OCCS[web]"  # installs fastapi, uvicorn, python-multipart
# or (option B)
pip install fastapi uvicorn python-multipart
```

Or with plain pip (Python >= 3.10):

```bash
pip install -e .
pip install numpy scipy scikit-learn scikit-optimize matplotlib
pip install "OCCS[web]"  # or: fastapi uvicorn python-multipart
```

### 2) Start the backend

```bash
occs-web --host 127.0.0.1 --port 8000
# Alternatively: python -m OCCS.service.app
```

The API will be available under `http://127.0.0.1:8000/api`.

### 3) Run the Web UI (dev or production)

Dev (hot reload via Vite proxy):

```bash
cd OCCS/webui
npm install  # or: pnpm install / yarn
npm run dev
# Open http://127.0.0.1:5173 (proxy to backend /api, including WS)
```

Production build (served by FastAPI):

```bash
cd OCCS/webui
npm run build
# Restart the backend; open http://127.0.0.1:8000/
```

### 4) Docker

Build locally:

```bash
docker build -t occs-web:local .
docker run --rm -p 8000:8000 occs-web:local
# Open http://127.0.0.1:8000/
```

Enable real hardware backend (optional):

```bash
docker run --rm -e OCCS_REAL_AVAILABLE=1 -p 8000:8000 occs-web:local
```

### 5) Using the app

- Select backend (mock by default). Real hardware is disabled unless `OCCS_REAL_AVAILABLE=1`.
- Configure wavelength grid and voltage bounds; optionally upload a target CSV.
- Create a session, adjust voltages manually, and refresh waveforms.
- Start optimization to stream loss and diagnostics; download history CSV anytime.

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
