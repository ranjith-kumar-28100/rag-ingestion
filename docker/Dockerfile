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
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        "torch==2.13.0+cpu" "torchvision==0.28.0+cpu" \
        -e ".[service]"

EXPOSE 8000 8501

# Overridden per service in docker-compose.
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
