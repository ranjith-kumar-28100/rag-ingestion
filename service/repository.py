"""DB access layer — the only module that reads/writes ORM rows.

API handlers and Celery tasks call these; neither touches the session directly.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .models import FileStatus, Job, JobFile, JobStatus
from .progress import compute_job_status, file_progress


@dataclass(frozen=True)
class NewFile:
    filename: str
    input_path: str
    file_type: str


def new_id() -> str:
    return uuid.uuid4().hex


def create_job(session: Session, job_id: str, output_dir: str, files: list[NewFile]) -> Job:
    """Persist a job and its per-file rows (all QUEUED)."""
    job = Job(
        id=job_id,
        status=JobStatus.QUEUED,
        total_files=len(files),
        output_dir=output_dir,
    )
    session.add(job)
    for spec in files:
        session.add(
            JobFile(
                id=new_id(),
                job_id=job_id,
                filename=spec.filename,
                file_type=spec.file_type,
                input_path=spec.input_path,
                status=FileStatus.QUEUED,
                progress=file_progress(FileStatus.QUEUED),
            )
        )
    session.flush()
    return job


def get_job(session: Session, job_id: str) -> Job | None:
    stmt = select(Job).where(Job.id == job_id).options(selectinload(Job.files))
    return session.execute(stmt).scalar_one_or_none()


def list_jobs(session: Session, limit: int = 50) -> list[Job]:
    stmt = (
        select(Job)
        .options(selectinload(Job.files))
        .order_by(Job.created_at.desc())
        .limit(limit)
    )
    return list(session.execute(stmt).scalars().all())


def get_file(session: Session, file_id: str) -> JobFile | None:
    return session.get(JobFile, file_id)


def list_job_files(session: Session, job_id: str) -> list[JobFile]:
    stmt = select(JobFile).where(JobFile.job_id == job_id).order_by(JobFile.created_at)
    return list(session.execute(stmt).scalars().all())


def mark_file_running(session: Session, file_id: str) -> None:
    jf = session.get(JobFile, file_id)
    if jf is None:
        return
    jf.status = FileStatus.RUNNING
    jf.progress = file_progress(FileStatus.RUNNING)
    _refresh_job_status(session, jf.job_id)


def mark_file_result(
    session: Session,
    file_id: str,
    *,
    status: FileStatus,
    output_path: str | None = None,
    chunk_count: int = 0,
    parent_count: int = 0,
    child_count: int = 0,
    total_tokens: int = 0,
    parent_strategy: str | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> None:
    jf = session.get(JobFile, file_id)
    if jf is None:
        return
    jf.status = status
    jf.progress = file_progress(status)
    jf.output_path = output_path
    jf.chunk_count = chunk_count
    jf.parent_count = parent_count
    jf.child_count = child_count
    jf.total_tokens = total_tokens
    jf.parent_strategy = parent_strategy
    jf.warnings = json.dumps(warnings) if warnings else None
    jf.error = error
    _refresh_job_status(session, jf.job_id)


def refresh_job_status(session: Session, job_id: str) -> None:
    _refresh_job_status(session, job_id)


def _refresh_job_status(session: Session, job_id: str) -> None:
    files = list_job_files(session, job_id)
    job = session.get(Job, job_id)
    if job is None:
        return
    job.status = compute_job_status([f.status for f in files])
