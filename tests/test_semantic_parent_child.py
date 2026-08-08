"""§9.1 / 9.2 / 9.4 / 9.8 — semantic parent-child chunking (pdf/docx)."""

from __future__ import annotations

from conftest import FakeEmbeddings, build_pdf_ir

from rag_ingestion.config import IngestionConfig
from rag_ingestion.models import BlockType, ChunkRole, IRDocument
from rag_ingestion.parsers.docling_parser import ParsedElement
from rag_ingestion.chunkers.semantic_parent_child import SemanticParentChildChunker


def _doc_with_table_and_text(cfg: IngestionConfig) -> IRDocument:
    els = [
        ParsedElement(kind="heading", text="1. Results", page=1, level=1),
        ParsedElement(kind="text", text="The revenue analysis is presented. " * 20, page=1),
        ParsedElement(
            kind="table",
            page=1,
            table_rows=[["Region", "Revenue"], ["North", "1200"], ["South", "SENTINEL_9999"]],
            table_markdown="| Region | Revenue |\n| --- | --- |\n| North | 1200 |\n| South | SENTINEL_9999 |",
            caption="Revenue by region",
        ),
    ]
    return build_pdf_ir(els, page_count=1, cfg=cfg)


def test_every_child_has_resolvable_parent(cfg: IngestionConfig, fake_embeddings: FakeEmbeddings) -> None:
    doc = _doc_with_table_and_text(cfg)
    chunks = SemanticParentChildChunker(fake_embeddings).chunk(doc, cfg)
    parent_ids = {c.chunk_id for c in chunks if c.role is ChunkRole.PARENT}
    children = [c for c in chunks if c.role is ChunkRole.CHILD]
    assert children
    for child in children:
        assert child.parent_id in parent_ids, f"orphan child {child.chunk_id}"


def test_no_table_content_in_text_chunks(cfg: IngestionConfig, fake_embeddings: FakeEmbeddings) -> None:
    doc = _doc_with_table_and_text(cfg)
    chunks = SemanticParentChildChunker(fake_embeddings).chunk(doc, cfg)
    text_chunks = [c for c in chunks if c.block_type is BlockType.TEXT]
    table_chunks = [c for c in chunks if c.block_type is BlockType.TABLE]
    assert any("SENTINEL_9999" in c.text for c in table_chunks)
    for c in text_chunks:
        assert "SENTINEL_9999" not in c.text


def test_table_row_group_split_repeats_header(cfg: IngestionConfig, fake_embeddings: FakeEmbeddings) -> None:
    # §9.4 — force a table split and assert the header appears in every fragment.
    small_cfg = cfg.model_copy(update={"max_child_tokens": 30})
    rows = [["ID", "Name", "Amount"]] + [[str(i), f"item{i}", str(i * 10)] for i in range(40)]
    md = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    els = [
        ParsedElement(kind="heading", text="Data", page=1, level=1),
        ParsedElement(kind="text", text="Table follows. " * 20, page=1),
        ParsedElement(kind="table", page=1, table_rows=rows, table_markdown=md),
    ]
    doc = build_pdf_ir(els, page_count=1, cfg=small_cfg)
    chunks = SemanticParentChildChunker(fake_embeddings).chunk(doc, small_cfg)
    fragments = [c for c in chunks if c.block_type is BlockType.TABLE]
    assert len(fragments) > 1, "table did not split"
    for frag in fragments:
        assert "ID" in frag.text and "Name" in frag.text and "Amount" in frag.text


def test_min_child_tokens_invariant(cfg: IngestionConfig, fake_embeddings: FakeEmbeddings) -> None:
    # §9.8 — no emitted child below threshold, except the single-child==parent case.
    tuned = cfg.model_copy(update={"min_child_tokens": 20, "max_child_tokens": 60})
    els = [
        ParsedElement(kind="heading", text="Body", page=1, level=1),
        ParsedElement(
            kind="text",
            text="\n\n".join(f"Paragraph {i} with a handful of words here." for i in range(30)),
            page=1,
        ),
    ]
    doc = build_pdf_ir(els, page_count=1, cfg=tuned)
    chunks = SemanticParentChildChunker(fake_embeddings).chunk(doc, tuned)

    parents = {c.chunk_id: c for c in chunks if c.role is ChunkRole.PARENT}
    for parent_id, parent in parents.items():
        siblings = [c for c in chunks if c.role is ChunkRole.CHILD and c.parent_id == parent_id]
        for child in siblings:
            single_equals_parent = len(siblings) == 1 and child.text == parent.text
            assert child.token_count >= tuned.min_child_tokens or single_equals_parent


def test_tiny_section_emits_parent_plus_identical_child(cfg: IngestionConfig, fake_embeddings: FakeEmbeddings) -> None:
    els = [
        ParsedElement(kind="heading", text="Tiny", page=1, level=1),
        ParsedElement(kind="text", text="Just a few words.", page=1),
    ]
    doc = build_pdf_ir(els, page_count=1, cfg=cfg)
    chunks = SemanticParentChildChunker(fake_embeddings).chunk(doc, cfg)
    parents = [c for c in chunks if c.role is ChunkRole.PARENT]
    children = [c for c in chunks if c.role is ChunkRole.CHILD]
    assert len(parents) == 1 and len(children) == 1
    assert children[0].text == parents[0].text
    assert children[0].parent_id == parents[0].chunk_id
