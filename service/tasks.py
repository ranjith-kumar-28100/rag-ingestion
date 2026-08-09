"""Celery tasks: one task per file, plus a chord callback to finalize the job.

The heavy ``IngestionPipeline`` (local embedding + parser models) is built once
per worker process and reused. Per-file failures never raise out of the task —
``ingest_file`` already isolates them — so one bad file cannot poison the batch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from celery import chord

from rag_ingestion import IngestionConfig, IngestionPipeline

from .celery_app import celery_app
from .config import get_settings
from .db import session_scope
from .models import FileStatus, Job
from .repository import mark_file_result, mark_file_running, refresh_job_status

logger = logging.getLogger("rag_ingestion.service.tasks")

_pipeline: IngestionPipeline | None = None


def get_pipeline() -> IngestionPipeline:
    """Lazily build one pipeline per worker process (models load once)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(IngestionConfig())
    return _pipeline


def _write_chunks(chunks: list[Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")


@celery_app.task(name="service.ingest_file", bind=True)  # type: ignore[untyped-decorator]
def ingest_file_task(self: Any, job_id: str, file_id: str, input_path: str, output_path: str) -> str:
    """Ingest a single file, write its chunks to JSONL, record stats. Never raises."""
    with session_scope() as session:
        mark_file_running(session, file_id)

    try:
        pipeline = get_pipeline()
        result = pipeline.ingest_file(input_path)
        _write_chunks(result.chunks, Path(output_path))

        status = {
            "success": FileStatus.DONE,
            "partial": FileStatus.PARTIAL,
            "failed": FileStatus.FAILED,
        }[result.status]

        with session_scope() as session:
            mark_file_result(
                session,
                file_id,
                status=status,
                output_path=output_path if result.chunks else None,
                chunk_count=result.stats.chunk_count,
                parent_count=result.stats.parent_count,
                child_count=result.stats.child_count,
                total_tokens=result.stats.total_tokens,
                parent_strategy=result.stats.parent_strategy,
                warnings=result.stats.warnings,
                error=result.error,
            )
        return status.value
    except Exception as exc:  # noqa: BLE001 - task must not crash the batch
        logger.exception("ingest_file_task failed job=%s file=%s", job_id, file_id)
        with session_scope() as session:
            mark_file_result(session, file_id, status=FileStatus.FAILED, error=str(exc))
        return FileStatus.FAILED.value


@celery_app.task(name="service.finalize_job")  # type: ignore[untyped-decorator]
def finalize_job(_results: list[str], job_id: str) -> str:
    """Chord callback: recompute the job status once all file tasks are done."""
    with session_scope() as session:
        refresh_job_status(session, job_id)
        job = session.get(Job, job_id)
        return job.status.value if job is not None else "unknown"


def enqueue_job(job_id: str, file_tasks: list[tuple[str, str, str]]) -> None:
    """Dispatch a chord of per-file tasks with a finalize callback.

    ``file_tasks`` is a list of ``(file_id, input_path, output_path)``.
    """
    settings = get_settings()  # noqa: F841 (kept for symmetry / future routing)
    header = [
        ingest_file_task.s(job_id, file_id, input_path, output_path)
        for file_id, input_path, output_path in file_tasks
    ]
    chord(header)(finalize_job.s(job_id))
