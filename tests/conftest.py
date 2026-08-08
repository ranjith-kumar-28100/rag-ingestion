"""Shared test fixtures and helpers.

Embeddings are always mocked — no live Azure calls in CI (spec section 9). Most
assertions run on synthetic ParsedElement lists so they never touch Docling.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from rag_ingestion.config import IngestionConfig
from rag_ingestion.models import IRDocument
from rag_ingestion.parsers.docling_parser import ParsedElement, resolve_strategy_and_build


class FakeEmbeddings(Embeddings):
    """Deterministic offline embeddings implementing the LangChain interface."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            out.append([digest[i % len(digest)] / 255.0 for i in range(self.dim)])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def cfg(tmp_path: Path) -> IngestionConfig:
    # cache disabled by default so tests are hermetic; enable per-test when needed
    return IngestionConfig(cache_dir=tmp_path / "cache", enable_embedding_cache=False)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    from generate_fixtures import FIXTURES, generate_all

    if not (FIXTURES / "multi_sheet.xlsx").exists():
        generate_all()
    return FIXTURES


def build_pdf_ir(
    elements: list[ParsedElement],
    page_count: int,
    cfg: IngestionConfig,
    *,
    file_type: str = "pdf",
    classifier: object | None = None,
    num_slides: int = 0,
) -> IRDocument:
    """Assemble an IRDocument straight from synthetic elements (no Docling)."""
    return resolve_strategy_and_build(
        "testdoc",
        "file:///test",
        file_type,
        elements,
        page_count,
        cfg,
        classifier=classifier,  # type: ignore[arg-type]
        num_slides=num_slides,
    )
