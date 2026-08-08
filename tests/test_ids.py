"""§9.5 — deterministic IDs (unit level)."""

from __future__ import annotations

from rag_ingestion.utils.ids import (
    compute_chunk_id,
    compute_doc_id,
    content_hash,
)


def test_doc_id_stable_for_same_inputs() -> None:
    a = compute_doc_id("file:///x.pdf", 1234.5)
    b = compute_doc_id("file:///x.pdf", 1234.5)
    assert a == b and len(a) == 16


def test_doc_id_changes_with_mtime() -> None:
    assert compute_doc_id("file:///x.pdf", 1.0) != compute_doc_id("file:///x.pdf", 2.0)


def test_chunk_id_stable_and_ordinal_sensitive() -> None:
    base = ("doc", ["A", "B"], "child", 0, "text")
    assert compute_chunk_id(*base) == compute_chunk_id(*base)
    assert compute_chunk_id("doc", ["A", "B"], "child", 0, "t") != compute_chunk_id(
        "doc", ["A", "B"], "child", 1, "t"
    )


def test_content_hash_stable() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")
