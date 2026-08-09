"""rag_ingestion microservice: FastAPI + Celery + SQLite around the pipeline.

Thin async wrapper over the `rag_ingestion` module. It adds job orchestration,
a message queue for graceful backpressure under heavy load, and progress
tracking — it never re-implements parsing or chunking. All processing stays on
the machine (the pipeline runs with the local providers / offline mode).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
