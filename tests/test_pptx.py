"""§9.1 (pptx) + §6.2 — slide chunking, parent windows, empty-slide skip."""

from __future__ import annotations

from conftest import FakeEmbeddings, build_pdf_ir

from rag_ingestion.config import IngestionConfig
from rag_ingestion.models import ChunkRole, IRDocument
from rag_ingestion.parsers.docling_parser import ParsedElement
from rag_ingestion.chunkers.slide import SlideChunker


def _deck(cfg: IngestionConfig) -> IRDocument:
    els: list[ParsedElement] = []
    for slide in range(1, 4):
        els.append(ParsedElement(kind="heading", text=f"Slide {slide} Title", page=slide, level=1))
        els.append(ParsedElement(kind="text", text=f"Body of slide {slide}. Refers to previous slide.", page=slide))
    # slide 4 is image-only (no elements) — declared via num_slides
    return build_pdf_ir(els, page_count=4, cfg=cfg, file_type="pptx", num_slides=4)


def test_pptx_children_have_resolvable_parents(cfg: IngestionConfig, fake_embeddings: FakeEmbeddings) -> None:
    doc = _deck(cfg)
    assert doc.raw_metadata["parent_strategy"] == "SLIDE"
    chunks = SlideChunker(fake_embeddings).chunk(doc, cfg)
    parent_ids = {c.chunk_id for c in chunks if c.role is ChunkRole.PARENT}
    children = [c for c in chunks if c.role is ChunkRole.CHILD]
    assert children
    for child in children:
        assert child.parent_id in parent_ids
        assert child.slide_number is not None


def test_empty_slide_skipped_and_counted(cfg: IngestionConfig, fake_embeddings: FakeEmbeddings) -> None:
    doc = _deck(cfg)
    chunks = SlideChunker(fake_embeddings).chunk(doc, cfg)
    assert doc.raw_metadata["empty_slides"] == 1
    # one child per non-empty slide (short slides aren't sub-split)
    children = [c for c in chunks if c.role is ChunkRole.CHILD]
    assert len(children) == 3


def test_parent_window_includes_neighbors(cfg: IngestionConfig, fake_embeddings: FakeEmbeddings) -> None:
    doc = _deck(cfg)
    chunks = SlideChunker(fake_embeddings).chunk(doc, cfg)
    # the parent for the middle slide's child should include neighbor slide text
    child2 = next(c for c in chunks if c.role is ChunkRole.CHILD and c.slide_number == 2)
    parent = next(c for c in chunks if c.chunk_id == child2.parent_id)
    assert "Slide 1 Title" in parent.text and "Slide 3 Title" in parent.text
