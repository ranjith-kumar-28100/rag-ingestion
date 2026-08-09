"""Celery task body: runs the pipeline (faked), writes JSONL, updates the DB.

The task is exercised via ``.apply()`` (eager, in-process) with a fake pipeline,
so no Redis/worker/models are needed but the real DB + serialization run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag_ingestion import IngestionResult, IngestionStats
from rag_ingestion.models import BlockType, Chunk, ChunkRole


class _FakePipeline:
    def ingest_file(self, path: str) -> IngestionResult:
        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            parent_id="p1",
            role=ChunkRole.CHILD,
            block_type=BlockType.TEXT,
            text="hello world",
            source_uri="file:///x",
            file_type="pdf",
            token_count=2,
        )
        return IngestionResult(
            doc_id="d1",
            source_uri="file:///x",
            status="success",
            chunks=[chunk],
            stats=IngestionStats(
                file_type="pdf",
                parse_seconds=0.01,
                chunk_count=1,
                parent_count=0,
                child_count=1,
                total_tokens=2,
                parent_strategy="HEADING",
            ),
        )


def test_ingest_file_task_writes_chunks_and_updates_db(
    service_db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from service import repository, tasks
    from service.db import session_scope
    from service.models import FileStatus, JobStatus

    monkeypatch.setattr(tasks, "get_pipeline", lambda: _FakePipeline())

    # seed a job + one file row
    job_id = repository.new_id()
    with session_scope() as session:
        job = repository.create_job(
            session,
            job_id,
            str(tmp_path),
            [repository.NewFile(filename="a.pdf", input_path="/nonexistent/a.pdf", file_type="pdf")],
        )
        file_id = job.files[0].id

    out_path = tmp_path / f"{file_id}.jsonl"
    result = tasks.ingest_file_task.apply(
        args=(job_id, file_id, "/nonexistent/a.pdf", str(out_path))
    ).get()
    assert result == "done"

    # output JSONL written with one chunk line
    assert out_path.exists()
    lines = [ln for ln in out_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1

    # DB row reflects the result; job rolled up to DONE
    with session_scope() as session:
        jf = repository.get_file(session, file_id)
        assert jf is not None
        assert jf.status is FileStatus.DONE
        assert jf.progress == 100
        assert jf.chunk_count == 1
        assert jf.parent_strategy == "HEADING"
        assert repository.get_job(session, job_id).status is JobStatus.DONE  # type: ignore[union-attr]


def test_ingest_file_task_handles_failure(
    service_db: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from service import repository, tasks
    from service.db import session_scope
    from service.models import FileStatus

    class _Boom:
        def ingest_file(self, path: str) -> Any:
            raise RuntimeError("kaboom")

    monkeypatch.setattr(tasks, "get_pipeline", lambda: _Boom())

    job_id = repository.new_id()
    with session_scope() as session:
        job = repository.create_job(
            session,
            job_id,
            str(tmp_path),
            [repository.NewFile(filename="a.pdf", input_path="/x/a.pdf", file_type="pdf")],
        )
        file_id = job.files[0].id

    result = tasks.ingest_file_task.apply(
        args=(job_id, file_id, "/x/a.pdf", str(tmp_path / "o.jsonl"))
    ).get()
    assert result == "failed"

    with session_scope() as session:
        jf = repository.get_file(session, file_id)
        assert jf is not None
        assert jf.status is FileStatus.FAILED
        assert jf.error is not None and "kaboom" in jf.error
