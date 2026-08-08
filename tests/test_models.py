"""§9.10 — to_langchain() metadata must be flat (no nested dicts/lists)."""

from __future__ import annotations

from rag_ingestion.models import BlockType, Chunk, ChunkRole


def _sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="c1",
        doc_id="d1",
        parent_id="p1",
        role=ChunkRole.CHILD,
        block_type=BlockType.TEXT,
        text="hello world",
        section_path=["3. Architecture", "3.2 Retrieval"],
        source_uri="file:///doc.pdf",
        file_type="pdf",
        page=4,
        row_range=(10, 19),
        token_count=2,
        metadata={"nested": {"a": 1}, "listy": [1, 2, 3], "flag": True, "table_id": "t1"},
    )


def test_to_langchain_metadata_is_flat() -> None:
    doc = _sample_chunk().to_langchain()
    assert doc.page_content == "hello world"
    for key, value in doc.metadata.items():
        assert isinstance(value, str | int | float | bool), f"{key} -> {type(value)} is not flat"


def test_section_path_is_joined_string() -> None:
    doc = _sample_chunk().to_langchain()
    assert doc.metadata["section_path"] == "3. Architecture > 3.2 Retrieval"


def test_row_range_split_into_ints() -> None:
    doc = _sample_chunk().to_langchain()
    assert doc.metadata["row_start"] == 10
    assert doc.metadata["row_end"] == 19


def test_parent_chunk_has_no_parent_id_key() -> None:
    parent = _sample_chunk().model_copy(update={"parent_id": None, "role": ChunkRole.PARENT})
    assert "parent_id" not in parent.to_langchain().metadata
