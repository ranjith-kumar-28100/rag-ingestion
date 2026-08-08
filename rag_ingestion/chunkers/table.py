"""Table chunking (spec section 6.3) — shared by every format that carries tables.

Each ``TableBlock`` becomes a standalone CHILD chunk (``block_type=TABLE``) whose
``parent_id`` is the parent of its enclosing section. Doc-level orphan tables get
a synthetic document-root parent. Oversized tables split by row groups, repeating
the header row in every fragment. Table content never reaches the semantic chunker.
"""

from __future__ import annotations

from ..config import IngestionConfig
from ..models import BlockType, Chunk, ChunkRole, IRDocument, TableBlock
from ..utils.ids import compute_chunk_id
from ..utils.tables import grid_to_markdown, split_table_by_rows
from ..utils.tokens import count_tokens


def _table_text(caption: str | None, markdown: str) -> str:
    return f"{caption}\n{markdown}" if caption else markdown


def build_table_chunks(
    doc: IRDocument,
    table: TableBlock,
    parent_id: str,
    cfg: IngestionConfig,
    ordinal: int,
) -> list[Chunk]:
    """Emit one or more CHILD chunks for a single table."""
    caption = table.caption
    full_text = _table_text(caption, table.markdown)

    if count_tokens(full_text) <= cfg.max_child_tokens or not table.rows:
        chunk = _mk_table_chunk(doc, table, parent_id, full_text, table.section_path, ordinal)
        return [chunk]

    # split by row groups, repeating the header in each fragment
    fragments = split_table_by_rows(table.rows, cfg.max_child_tokens, has_header=True)
    chunks: list[Chunk] = []
    n = len(fragments)
    for i, frag_md in enumerate(fragments):
        text = _table_text(caption, frag_md)
        path = table.section_path + [f"(table part {i + 1}/{n})"]
        chunks.append(_mk_table_chunk(doc, table, parent_id, text, path, ordinal + i))
    return chunks


def _mk_table_chunk(
    doc: IRDocument,
    table: TableBlock,
    parent_id: str,
    text: str,
    section_path: list[str],
    ordinal: int,
) -> Chunk:
    return Chunk(
        chunk_id=compute_chunk_id(doc.doc_id, section_path, "child", ordinal, text),
        doc_id=doc.doc_id,
        parent_id=parent_id,
        role=ChunkRole.CHILD,
        block_type=BlockType.TABLE,
        text=text,
        section_path=section_path,
        source_uri=doc.source_uri,
        file_type=doc.file_type,
        page=table.page,
        token_count=count_tokens(text),
        metadata={"table_id": table.table_id},
    )


def synthetic_root_parent(doc: IRDocument) -> Chunk:
    """Document-root parent for orphan (doc-level) tables."""
    text = doc.title or doc.doc_id
    return Chunk(
        chunk_id=compute_chunk_id(doc.doc_id, [], "parent", -1, "document-root"),
        doc_id=doc.doc_id,
        parent_id=None,
        role=ChunkRole.PARENT,
        block_type=BlockType.TEXT,
        text=text,
        section_path=[],
        source_uri=doc.source_uri,
        file_type=doc.file_type,
        token_count=count_tokens(text),
        metadata={"synthetic_root": True},
    )


class TableChunker:
    """Standalone table chunker (used directly for table-only IR / tests)."""

    file_types: set[str] = set()

    def chunk(self, doc: IRDocument, cfg: IngestionConfig) -> list[Chunk]:
        from ..utils.tokens import configure_token_counter

        configure_token_counter(cfg)
        root = synthetic_root_parent(doc)
        chunks: list[Chunk] = [root]
        ordinal = 0
        for section in doc.sections:
            for table in section.tables:
                if not table.markdown and table.rows:
                    table.markdown = grid_to_markdown(table.rows)
                new = build_table_chunks(doc, table, root.chunk_id, cfg, ordinal)
                chunks.extend(new)
                ordinal += len(new)
        for table in doc.tables:
            new = build_table_chunks(doc, table, root.chunk_id, cfg, ordinal)
            chunks.extend(new)
            ordinal += len(new)
        return chunks
