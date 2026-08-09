"""FastAPI app: submit batch ingest jobs, poll progress, fetch chunks.

Endpoints:
    POST /jobs                 multipart upload -> {id, status, total_files}
    GET  /jobs                 recent jobs (summaries)
    GET  /jobs/{id}            job + per-file progress (percent)
    GET  /jobs/{id}/chunks     produced chunks (JSONL), optional ?file_id
    GET  /health

The API only orchestrates: it saves uploads, records rows, and enqueues Celery
tasks. All parsing/chunking happens in workers, so a large upload returns a
job id immediately instead of blocking the request.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from . import repository, tasks
from .config import get_settings
from .db import SessionLocal, init_db
from .schemas import JobCreatedOut, JobOut, JobSummaryOut

# Mirrors rag_ingestion's parser registry. Kept as a literal so the API does not
# import Docling/torch just to validate an extension.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".csv"}
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    init_db()
    yield


app = FastAPI(title="rag_ingestion service", version="0.1.0", lifespan=lifespan)


def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", response_model=JobCreatedOut, status_code=202)
async def create_job(
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
) -> JobCreatedOut:
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")

    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    job_id = repository.new_id()

    input_dir = settings.upload_dir / job_id
    output_dir = settings.output_dir / job_id
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs: list[repository.NewFile] = []
    file_ids: list[str] = []
    for upload in files:
        name = Path(upload.filename or "upload").name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported file type {ext!r}; supported: {sorted(SUPPORTED_EXTENSIONS)}",
            )
        data = await upload.read()
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"{name} exceeds {settings.max_upload_mb} MB limit"
            )
        file_id = repository.new_id()
        dest = input_dir / f"{file_id}{ext}"
        dest.write_bytes(data)
        specs.append(
            repository.NewFile(filename=name, input_path=str(dest), file_type=ext.lstrip("."))
        )
        file_ids.append(file_id)

    job = repository.create_job(session, job_id, str(output_dir), specs)
    # rows are created in declared order; pair each with its pre-generated id
    file_tasks = [
        (jf.id, jf.input_path, str(output_dir / f"{jf.id}.jsonl"))
        for jf in job.files
    ]
    session.commit()

    tasks.enqueue_job(job_id, file_tasks)
    return JobCreatedOut(id=job.id, status=job.status.value, total_files=job.total_files)


@app.get("/jobs", response_model=list[JobSummaryOut])
def list_jobs(
    limit: int = Query(50, ge=1, le=500),
    session: Session = Depends(get_session),
) -> list[JobSummaryOut]:
    return [JobSummaryOut.from_row(j) for j in repository.list_jobs(session, limit=limit)]


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, session: Session = Depends(get_session)) -> JobOut:
    job = repository.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobOut.from_row(job)


@app.get("/jobs/{job_id}/chunks")
def get_chunks(
    job_id: str,
    file_id: str | None = Query(None),
    limit: int = Query(1000, ge=1, le=100_000),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    job = repository.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    rows = [f for f in job.files if f.output_path]
    if file_id is not None:
        rows = [f for f in rows if f.id == file_id]
        if not rows:
            raise HTTPException(status_code=404, detail="file not found or produced no chunks")

    chunks: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row.output_path) if row.output_path else None
        if path is None or not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

    total = len(chunks)
    page = chunks[offset : offset + limit]
    return {"job_id": job_id, "total": total, "offset": offset, "limit": limit, "chunks": page}
