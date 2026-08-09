"""API flow: upload -> job rows -> (simulated worker) -> poll -> fetch chunks.

Celery/Redis/the real pipeline are replaced by a fake ``enqueue_job`` that does
what a worker would: write a chunk JSONL and mark the file done. This keeps the
test hermetic while exercising the API, DB, schemas, and chunk endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(service_db: object, monkeypatch: pytest.MonkeyPatch) -> Any:
    from service import api, repository, tasks
    from service.db import session_scope
    from service.models import FileStatus

    def fake_enqueue(job_id: str, file_tasks: list[tuple[str, str, str]]) -> None:
        for file_id, _input_path, output_path in file_tasks:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"chunk_id": file_id, "text": "hello", "role": "child"}) + "\n")
            with session_scope() as session:
                repository.mark_file_result(
                    session,
                    file_id,
                    status=FileStatus.DONE,
                    output_path=str(out),
                    chunk_count=1,
                    parent_count=0,
                    child_count=1,
                    total_tokens=5,
                    parent_strategy="HEADING",
                )

    monkeypatch.setattr(tasks, "enqueue_job", fake_enqueue)
    with TestClient(api.app) as c:
        yield c


def test_health(client: Any) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_full_job_flow(client: Any) -> None:
    files = [
        ("files", ("a.pdf", b"%PDF-1.4 fake", "application/pdf")),
        ("files", ("b.csv", b"x,y\n1,2\n", "text/csv")),
    ]
    resp = client.post("/jobs", files=files)
    assert resp.status_code == 202
    created = resp.json()
    assert created["total_files"] == 2
    job_id = created["id"]

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["progress"] == 100
    assert job["files_done"] == 2
    assert {f["filename"] for f in job["files"]} == {"a.pdf", "b.csv"}
    assert all(f["parent_strategy"] == "HEADING" for f in job["files"])

    chunks = client.get(f"/jobs/{job_id}/chunks").json()
    assert chunks["total"] == 2
    assert len(chunks["chunks"]) == 2

    # per-file filter
    fid = job["files"][0]["id"]
    one = client.get(f"/jobs/{job_id}/chunks", params={"file_id": fid}).json()
    assert one["total"] == 1


def test_unsupported_extension_rejected(client: Any) -> None:
    resp = client.post("/jobs", files=[("files", ("bad.txt", b"nope", "text/plain"))])
    assert resp.status_code == 415


def test_missing_job_404(client: Any) -> None:
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/jobs/does-not-exist/chunks").status_code == 404


def test_job_list(client: Any) -> None:
    client.post("/jobs", files=[("files", ("a.csv", b"x\n1\n", "text/csv"))])
    listing = client.get("/jobs").json()
    assert len(listing) >= 1
    assert {"id", "status", "progress", "total_files"} <= set(listing[0])
