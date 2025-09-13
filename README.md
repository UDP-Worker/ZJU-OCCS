# ZJU-OCCS: Optical Chip Control & Search

[English](./docs/README.en.md) | [简体中文](./docs/README.zh-CN.md)

OCCS is a front-end + back-end solution to control optical hardware and optimize voltages so the measured spectrum matches a target curve. It ships a React web UI, a FastAPI service with REST/WS, and reusable core modules for simulation and Bayesian optimization.

Key highlights
- Live web UI for waveform and loss with manual voltage control
- FastAPI service with sessions, CSV export, and real-time streaming
- Core, test-backed modules for objectives and skopt-based BO

Quick start
1) Create environment
```bash
mamba env create -f environment.yml && mamba activate ZJU-OCCS   # or conda
pip install fastapi uvicorn                                      # for the web service
```

2) Start backend
```bash
occs-web --host 127.0.0.1 --port 8000
```

3) Run the web UI
```bash
cd OCCS/webui && npm install && npm run dev   # dev proxy on :5173
# or: npm run build and open http://127.0.0.1:8000/ (served by FastAPI)
```

More docs (architecture, API, and workflows): see the language-specific READMEs above.

License
This project is licensed under the terms of the LICENSE file in this repository.
