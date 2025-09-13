# --- Stage 1: build the Web UI ---
FROM node:20-bullseye AS webui-builder

WORKDIR /work

# Copy only the webui to leverage Docker layer caching effectively
COPY OCCS/webui/package.json OCCS/webui/package-lock.json ./OCCS/webui/
RUN --mount=type=cache,target=/root/.npm \
    cd OCCS/webui && npm ci

COPY OCCS/webui ./OCCS/webui
RUN cd OCCS/webui && npm run build

# --- Stage 2: runtime image with Python service ---
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OCCS_REAL_AVAILABLE=0

WORKDIR /app

# System deps (optional but useful for scientific wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project (without node_modules) and webui build artifacts
COPY . .
COPY --from=webui-builder /work/OCCS/webui/dist ./OCCS/webui/dist

# Install package with web extras (installs fastapi + uvicorn)
RUN python -m pip install --upgrade pip \
    && pip install -e .[web]

EXPOSE 8000

ENTRYPOINT ["occs-web", "--host", "0.0.0.0", "--port", "8000"]
