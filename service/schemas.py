"""API response DTOs (Pydantic). Kept separate from ORM rows on purpose."""

from __future__ import annotations

import json
from datetime import datetime

from pydantic import BaseModel

from .models import Job, JobFile
from .progress import compute_job_progress


class JobFileOut(BaseModel):
    id: str
    filename: str
    file_type: str
    status: str
    progress: int
    chunk_count: int
    parent_count: int
    child_count: int
    total_tokens: int
    parent_strategy: str | None
    warnings: list[str]
    error: str | None

    @classmethod
    def from_row(cls, row: JobFile) -> "JobFileOut":
        return cls(
            id=row.id,
            filename=row.filename,
            file_type=row.file_type,
            status=row.status.value,
            progress=row.progress,
            chunk_count=row.chunk_count,
            parent_count=row.parent_count,
            child_count=row.child_count,
            total_tokens=row.total_tokens,
            parent_strategy=row.parent_strategy,
            warnings=json.loads(row.warnings) if row.warnings else [],
            error=row.error,
        )


class JobOut(BaseModel):
    id: str
    status: str
    progress: int
    total_files: int
    files_done: int
    files_failed: int
    created_at: datetime
    updated_at: datetime
    error: str | None
    files: list[JobFileOut]

    @classmethod
    def from_row(cls, job: Job) -> "JobOut":
        files = list(job.files)
        return cls(
            id=job.id,
            status=job.status.value,
            progress=compute_job_progress([f.status for f in files]),
            total_files=job.total_files,
            files_done=sum(1 for f in files if f.status.value in ("done", "partial")),
            files_failed=sum(1 for f in files if f.status.value == "failed"),
            created_at=job.created_at,
            updated_at=job.updated_at,
            error=job.error,
            files=[JobFileOut.from_row(f) for f in files],
        )


class JobSummaryOut(BaseModel):
    """Lightweight row for the job list (no per-file detail)."""

    id: str
    status: str
    progress: int
    total_files: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, job: Job) -> "JobSummaryOut":
        files = list(job.files)
        return cls(
            id=job.id,
            status=job.status.value,
            progress=compute_job_progress([f.status for f in files]),
            total_files=job.total_files,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


class JobCreatedOut(BaseModel):
    id: str
    status: str
    total_files: int
