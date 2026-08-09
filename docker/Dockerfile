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

# Pin torch + torchvision to matched "+cpu" builds in the SAME resolution as the
# project. The "+cpu" local version exists ONLY on the PyTorch cpu index, so uv
# cannot substitute a CUDA torchvision from PyPI — that substitution is what
# broke torchvision::nms against CPU torch. --index-strategy unsafe-best-match
# lets uv pull these two from the extra index while everything else resolves
# from PyPI. CPU wheels also avoid the multi-GB CUDA download unused on CPU/WSL2.
# uv --no-cache is REQUIRED here, not an optimization. torchvision publishes the
# CUDA build to PyPI with plain metadata version "0.28.0" while the cpu index has
# "0.28.0+cpu". uv's cache keys the unpacked wheel loosely enough that, once the
# CUDA wheel is cached, it LINKS that stale wheel even when resolution correctly
# selects 0.28.0+cpu — silently reintroducing the torchvision::nms crash against
# CPU torch. Disabling the cache forces the right wheel to be fetched and linked
# every time. (Verified: with the cache, on-disk torchvision is 0.28.0+cu130
# despite the resolver reporting +cpu; with --no-cache it is 0.28.0+cpu.)
RUN uv pip install --system --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        "torch==2.13.0+cpu" "torchvision==0.28.0+cpu" \
        -e ".[service]"

EXPOSE 8000 8501

# Overridden per service in docker-compose.
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
