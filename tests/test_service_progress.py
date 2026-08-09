"""Pure progress/status roll-up logic (no DB, no Celery)."""

from __future__ import annotations

from service.models import FileStatus, JobStatus
from service.progress import compute_job_progress, compute_job_status


def test_progress_empty() -> None:
    assert compute_job_progress([]) == 0
    assert compute_job_status([]) is JobStatus.QUEUED


def test_progress_is_mean_of_files() -> None:
    # one done (100), one running (50), one queued (0) -> mean 50
    statuses = [FileStatus.DONE, FileStatus.RUNNING, FileStatus.QUEUED]
    assert compute_job_progress(statuses) == 50


def test_progress_full() -> None:
    assert compute_job_progress([FileStatus.DONE, FileStatus.PARTIAL]) == 100


def test_status_running_when_any_in_flight() -> None:
    assert compute_job_status([FileStatus.RUNNING, FileStatus.QUEUED]) is JobStatus.RUNNING
    assert compute_job_status([FileStatus.DONE, FileStatus.QUEUED]) is JobStatus.RUNNING


def test_status_queued_when_all_queued() -> None:
    assert compute_job_status([FileStatus.QUEUED, FileStatus.QUEUED]) is JobStatus.QUEUED


def test_status_done_when_all_done() -> None:
    assert compute_job_status([FileStatus.DONE, FileStatus.DONE]) is JobStatus.DONE


def test_status_partial_when_mixed_terminal() -> None:
    assert compute_job_status([FileStatus.DONE, FileStatus.FAILED]) is JobStatus.PARTIAL
    assert compute_job_status([FileStatus.DONE, FileStatus.PARTIAL]) is JobStatus.PARTIAL


def test_status_failed_when_all_failed() -> None:
    assert compute_job_status([FileStatus.FAILED, FileStatus.FAILED]) is JobStatus.FAILED
