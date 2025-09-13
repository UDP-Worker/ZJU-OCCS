# ZJU-OCCS: Optical Chip Control & Search

[English](./docs/README.en.md) | [简体中文](./docs/README.zh-CN.md)

OCCS is a front-end + back-end solution to control optical hardware and optimize voltages so the measured spectrum matches a target curve. It ships a React web UI, a FastAPI service with REST/WS, and reusable core modules for simulation and Bayesian optimization.

Key highlights
- Live web UI for waveform and loss with manual voltage control
- FastAPI service with sessions, CSV export, and real-time streaming
- Core, test-backed modules for objectives and skopt-based BO

Quick start (Docker, recommended)
- Download the prebuilt image tarball from the GitHub Release assets, e.g. `occs-web.tar.gz`.
- Load it: `docker load -i occs-web.tar.gz` (note the image name:tag printed).
- Run: `docker run --rm -p 8000:8000 <image:tag>` then open `http://127.0.0.1:8000/`.
- Optional real hardware: add `-e OCCS_REAL_AVAILABLE=1`.
- Optional persistence: add `-v $(pwd)/uploads:/tmp/occs_uploads` to keep uploaded target CSVs.

More docs (architecture, API, and workflows): see the language-specific READMEs above.

License
This project is licensed under the terms of the LICENSE file in this repository.
