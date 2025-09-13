# Release Guide

This document describes how to produce a versioned release of OCCS (Python package + built Web UI) and an optional Docker image.

## 1. Bump version

- Edit `pyproject.toml` → `[project].version` (semantic versioning), e.g. `0.1.0` → `0.1.1`.

## 2. Build the Web UI

```bash
cd OCCS/webui
npm ci
npm run build  # outputs OCCS/webui/dist
```

## 3. Run tests

```bash
pytest -q
```

## 4. Build Python artifacts

```bash
python -m pip install --upgrade build
python -m build  # creates dist/*.tar.gz and dist/*.whl
```

Verify the wheel includes `OCCS/webui/dist/**` and `OCCS/data/*.csv` (already configured via `pyproject.toml` + `MANIFEST.in`).

Optional: smoke-test the wheel in a fresh venv:

```bash
python -m venv .venv-test && . .venv-test/bin/activate
pip install dist/OCCS-<ver>-py3-none-any.whl
pip install "OCCS[web]"  # or: fastapi uvicorn python-multipart
occs-web --host 127.0.0.1 --port 8000
```

## 5. Publish

### PyPI (optional)

```bash
python -m pip install --upgrade twine
twine upload dist/*
```

### Git tags & GitHub Release

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push --tags
# Create a GitHub Release for the tag and attach dist/*.whl and *.tar.gz
```

## 6. Docker image

Build and run locally:

```bash
docker build -t occs-web:X.Y.Z .
docker run --rm -p 8000:8000 occs-web:X.Y.Z
```

Optional registry push (example):

```bash
docker tag occs-web:X.Y.Z ghcr.io/<org>/occs-web:X.Y.Z
docker push ghcr.io/<org>/occs-web:X.Y.Z
```

## 7. Release notes checklist

- Features & fixes since last tag
- How to run (pip + occs-web; Docker run command)
- Environment variable for real hardware: `OCCS_REAL_AVAILABLE=1`
- API compatibility notes (if any)
