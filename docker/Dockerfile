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

# Install the CPU-only torch build FIRST. The default PyPI torch wheel drags in
# multi-GB NVIDIA CUDA libraries we never use on CPU/WSL2; the cpu index is a
# fraction of the size and download time. sentence-transformers/docling then
# see torch as already satisfied. The cache mount keeps wheels between builds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

# Project metadata + source. Editable install, so code edits don't need a
# reinstall; and with the uv cache mount a rebuild re-links from cache instead
# of re-downloading the heavy wheels (docling, torch, transformers, ...).
COPY pyproject.toml README.md ./
COPY rag_ingestion ./rag_ingestion
COPY service ./service
COPY ui ./ui
COPY alembic ./alembic
COPY alembic.ini ./
COPY docker ./docker

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e ".[service]"

EXPOSE 8000 8501

# Overridden per service in docker-compose.
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
