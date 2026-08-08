"""End-to-end Docling parsing on the real fixtures.

Skipped by default: Docling downloads OCR/TableFormer models on first run, which
is unsuitable for a fast/offline CI. Enable with RAG_INGEST_RUN_DOCLING=1.
Embeddings are still mocked (no live Azure).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import FakeEmbeddings

from rag_ingestion.config import IngestionConfig
from rag_ingestion.models import BlockType, ChunkRole
from rag_ingestion.parsers.docling_parser import DoclingParser
from rag_ingestion.pipeline import IngestionPipeline

pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_INGEST_RUN_DOCLING") != "1",
    reason="set RAG_INGEST_RUN_DOCLING=1 to run Docling integration tests",
)


@pytest.fixture
def cfg(tmp_path: Path) -> IngestionConfig:
    return IngestionConfig(cache_dir=tmp_path / "cache", enable_embedding_cache=False)


def test_multi_heading_pdf_end_to_end(fixtures_dir: Path, cfg: IngestionConfig) -> None:
    pipe = IngestionPipeline(cfg, embeddings=FakeEmbeddings())
    result = pipe.ingest_file(fixtures_dir / "multi_heading.pdf")
    assert result.status in ("success", "partial")
    assert result.stats.parent_strategy in ("HEADING", "PAGE")
    parent_ids = {c.chunk_id for c in result.chunks if c.role is ChunkRole.PARENT}
    for child in (c for c in result.chunks if c.role is ChunkRole.CHILD):
        assert child.parent_id in parent_ids
    # table content isolated from TEXT chunks
    assert any(c.block_type is BlockType.TABLE for c in result.chunks)


def test_flat_pdf_resolves_to_page(fixtures_dir: Path, cfg: IngestionConfig) -> None:
    doc = DoclingParser(cfg)
    ir = doc.parse(fixtures_dir / "flat.pdf")
    assert ir.raw_metadata["parent_strategy"] in ("PAGE", "HEADING")


def test_headings_docx_end_to_end(fixtures_dir: Path, cfg: IngestionConfig) -> None:
    pipe = IngestionPipeline(cfg, embeddings=FakeEmbeddings())
    result = pipe.ingest_file(fixtures_dir / "headings.docx")
    assert result.status in ("success", "partial")
    assert result.chunks
