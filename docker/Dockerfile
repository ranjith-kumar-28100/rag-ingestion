# Single image shared by api / worker / ui / prefetch (command differs per service).
# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# libgl1/libglib2.0-0: Docling/OpenCV runtime deps. curl: healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Project metadata + source. Editable install, so code edits don't need a
# reinstall; the uv cache mount re-links heavy wheels from cache on rebuild.
COPY pyproject.toml README.md ./
COPY rag_ingestion ./rag_ingestion
COPY service ./service
COPY ui ./ui
COPY alembic ./alembic
COPY alembic.ini ./
COPY docker ./docker

# --torch-backend=cpu resolves torch + torchvision + the whole graph in ONE
# pass, pulling matched CPU wheels from the PyTorch index. A two-phase install
# (cpu torch first, then the project) let the second resolution swap in a
# mismatched torchvision, whose ops (torchvision::nms) fail to bind to CPU
# torch. One unified resolution keeps the pair consistent — and CPU wheels
# avoid the multi-GB CUDA download we never use on CPU/WSL2.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --torch-backend=cpu -e ".[service]"

EXPOSE 8000 8501

# Overridden per service in docker-compose.
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
