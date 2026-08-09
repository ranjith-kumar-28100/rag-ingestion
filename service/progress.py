"""Pure functions for job-level progress and status roll-up.

Kept side-effect free so they are trivial to unit test without a DB.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import TERMINAL_FILE_STATUSES, FileStatus, JobStatus

# Per-file progress checkpoints. A single file is one atomic pipeline call, so
# its progress is coarse; the smooth job-level percent comes from averaging many
# files. (Users chose percent progress; batch averaging is what makes it move.)
FILE_PROGRESS: dict[FileStatus, int] = {
    FileStatus.QUEUED: 0,
    FileStatus.RUNNING: 50,
    FileStatus.DONE: 100,
    FileStatus.PARTIAL: 100,
    FileStatus.FAILED: 100,
}


def file_progress(status: FileStatus) -> int:
    return FILE_PROGRESS[status]


def compute_job_progress(file_statuses: Sequence[FileStatus]) -> int:
    """Job percent = mean of per-file progress (0 when empty)."""
    if not file_statuses:
        return 0
    return round(sum(FILE_PROGRESS[s] for s in file_statuses) / len(file_statuses))


def compute_job_status(file_statuses: Sequence[FileStatus]) -> JobStatus:
    """Roll per-file statuses up into one job status."""
    if not file_statuses:
        return JobStatus.QUEUED

    all_terminal = all(s in TERMINAL_FILE_STATUSES for s in file_statuses)
    if not all_terminal:
        any_started = any(s is not FileStatus.QUEUED for s in file_statuses)
        return JobStatus.RUNNING if any_started else JobStatus.QUEUED

    if all(s is FileStatus.FAILED for s in file_statuses):
        return JobStatus.FAILED
    if any(s in (FileStatus.FAILED, FileStatus.PARTIAL) for s in file_statuses):
        return JobStatus.PARTIAL
    return JobStatus.DONE
