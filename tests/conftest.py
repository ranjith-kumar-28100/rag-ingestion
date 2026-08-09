"""Shared test fixtures and helpers.

Embeddings are always mocked — no live Azure calls in CI (spec section 9). Most
assertions run on synthetic ParsedElement lists so they never touch Docling.
"""

from __future__ import annotations

import os

# Point the microservice at a throwaway DB/dirs BEFORE service modules import
# (service.db builds its engine at import time). Harmless for non-service tests.
os.environ.setdefault("RAG_SVC_DATABASE_URL", "sqlite:///./data/test_rag_service.db")
os.environ.setdefault("RAG_SVC_DATA_DIR", "./data/test")
os.environ.setdefault("RAG_SVC_UPLOAD_DIR", "./data/test/uploads")
os.environ.setdefault("RAG_SVC_OUTPUT_DIR", "./data/test/output")

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
    # cache disabled + tiktoken pinned so tests are hermetic and fast; the HF
    # token backend and auto-selection are covered directly in test_tokens.py.
    return IngestionConfig(
        cache_dir=tmp_path / "cache",
        enable_embedding_cache=False,
        token_counter="tiktoken",
    )


@pytest.fixture
def service_db() -> object:
    """Fresh service DB schema per test (drop + create on the shared engine)."""
    from service import models  # noqa: F401  (register tables on Base.metadata)
    from service.db import Base, engine

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield None
    Base.metadata.drop_all(engine)


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
