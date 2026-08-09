"""ORM models: a ``Job`` owns many ``JobFile`` rows (batch ingest).

Only progress/bookkeeping lives in the DB. The produced chunks are written to
JSONL on disk (``JobFile.output_path``); the DB stays small and the row shape
never changes with the chunk schema.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    PARTIAL = "partial"  # finished, but some files failed or degraded
    FAILED = "failed"  # every file failed


class FileStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    PARTIAL = "partial"  # produced chunks but with warnings (e.g. degraded parents)
    FAILED = "failed"


TERMINAL_FILE_STATUSES: frozenset[FileStatus] = frozenset(
    {FileStatus.DONE, FileStatus.PARTIAL, FileStatus.FAILED}
)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.QUEUED, nullable=False, index=True
    )
    total_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    files: Mapped[list["JobFile"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="JobFile.created_at",
    )


class JobFile(Base):
    __tablename__ = "job_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    input_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    output_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    status: Mapped[FileStatus] = mapped_column(
        Enum(FileStatus), default=FileStatus.QUEUED, nullable=False, index=True
    )
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 0..100

    # stats mirrored from IngestionResult for cheap listing without reading JSONL
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parent_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    child_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parent_strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)
    warnings: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    job: Mapped["Job"] = relationship(back_populates="files")
