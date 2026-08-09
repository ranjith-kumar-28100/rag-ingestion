# Single image shared by api / worker / ui / prefetch (command differs per service).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# libgl1/libglib2.0-0: Docling/OpenCV runtime deps. curl: healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install deps first (better layer caching), then the source.
COPY pyproject.toml README.md ./
COPY rag_ingestion ./rag_ingestion
COPY service ./service
COPY ui ./ui
COPY alembic ./alembic
COPY alembic.ini ./
COPY docker ./docker

RUN uv pip install --system -e ".[service]"

EXPOSE 8000 8501

# Overridden per service in docker-compose.
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
