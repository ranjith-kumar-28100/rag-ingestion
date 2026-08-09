"""Small Streamlit UI for the rag_ingestion service.

Upload files -> submit a batch job -> watch percent progress -> browse/download
the produced chunks. Talks to the FastAPI service over HTTP only; it holds no
ingestion logic itself.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import streamlit as st

API_BASE = os.environ.get("RAG_SVC_API_BASE_URL", "http://localhost:8000")
SUPPORTED = ["pdf", "docx", "pptx", "xlsx", "csv"]

st.set_page_config(page_title="rag_ingestion", page_icon="📄", layout="wide")


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=30.0)


def submit_job(files: list[Any]) -> dict[str, Any]:
    multipart = [("files", (f.name, f.getvalue())) for f in files]
    with _client() as client:
        resp = client.post("/jobs", files=multipart)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data


def fetch_job(job_id: str) -> dict[str, Any]:
    with _client() as client:
        resp = client.get(f"/jobs/{job_id}")
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data


def fetch_chunks(job_id: str, limit: int = 200) -> dict[str, Any]:
    with _client() as client:
        resp = client.get(f"/jobs/{job_id}/chunks", params={"limit": limit})
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data


def service_up() -> bool:
    try:
        with _client() as client:
            return client.get("/health").status_code == 200
    except httpx.HTTPError:
        return False


# --------------------------------------------------------------------------- #
st.title("📄 rag_ingestion")
st.caption("Local document ingestion & chunking — nothing leaves this machine.")

if not service_up():
    st.error(f"API not reachable at {API_BASE}. Start it (see README) and reload.")
    st.stop()

with st.sidebar:
    st.header("New job")
    uploads = st.file_uploader(
        "Documents", type=SUPPORTED, accept_multiple_files=True
    )
    if st.button("Ingest", type="primary", disabled=not uploads):
        created = submit_job(uploads or [])
        st.session_state["job_id"] = created["id"]
        st.success(f"Job {created['id'][:8]} queued ({created['total_files']} file(s))")

    manual = st.text_input("…or open an existing job id")
    if manual:
        st.session_state["job_id"] = manual.strip()

job_id = st.session_state.get("job_id")
if not job_id:
    st.info("Upload one or more documents and press **Ingest**.")
    st.stop()

job = fetch_job(job_id)
status = job["status"]

st.subheader(f"Job `{job_id[:12]}` — {status}")
st.progress(job["progress"] / 100.0, text=f"{job['progress']}%")

col1, col2, col3 = st.columns(3)
col1.metric("Files", job["total_files"])
col2.metric("Done", job["files_done"])
col3.metric("Failed", job["files_failed"])

st.markdown("#### Files")
st.dataframe(
    [
        {
            "file": f["filename"],
            "type": f["file_type"],
            "status": f["status"],
            "progress": f"{f['progress']}%",
            "chunks": f["chunk_count"],
            "parents": f["parent_count"],
            "children": f["child_count"],
            "strategy": f["parent_strategy"] or "",
            "warnings": "; ".join(f["warnings"]),
            "error": f["error"] or "",
        }
        for f in job["files"]
    ],
    use_container_width=True,
    hide_index=True,
)

is_terminal = status in ("done", "partial", "failed")

if is_terminal:
    if any(f["chunk_count"] for f in job["files"]):
        st.markdown("#### Chunks")
        payload = fetch_chunks(job_id)
        st.caption(f"Showing {len(payload['chunks'])} of {payload['total']} chunks")
        jsonl = "\n".join(json.dumps(c, ensure_ascii=False) for c in payload["chunks"])
        st.download_button(
            "Download chunks (.jsonl)", jsonl, file_name=f"{job_id}.jsonl", mime="application/json"
        )
        with st.expander("Preview"):
            st.json(payload["chunks"][:20])
else:
    time.sleep(1.5)
    st.rerun()
