"""PDF / DOCX chunker — semantic children under heading/page-anchored parents
(spec section 6.1).

Parents come straight from ``IRDocument.sections`` (however the parser resolved
them). Children are semantic splits *within* a parent, so ``parent_id`` mapping is
exact. Small children merge into the previous sibling; oversized children re-split
with the recursive splitter. Tables are delegated to the shared table chunker and
never touch the semantic splitter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..config import IngestionConfig
from ..models import BlockType, Chunk, ChunkRole, IRDocument, Section
from ..utils.ids import compute_chunk_id
from ..utils.tokens import count_tokens
from .table import build_table_chunks, synthetic_root_parent

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def _split_by_paragraphs(text: str, max_tokens: int) -> list[str]:
    """Group paragraphs into blocks each <= max_tokens (never splits a paragraph)."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    cur: list[str] = []
    for para in paras:
        if cur and count_tokens("\n\n".join(cur + [para])) > max_tokens:
            out.append("\n\n".join(cur))
            cur = [para]
        else:
            cur.append(para)
    if cur:
        out.append("\n\n".join(cur))
    return out or [text]


def _merge_small(pieces: list[str], min_tokens: int) -> list[str]:
    """Merge any child < min_tokens into the previous sibling (spec 6.1 step 4)."""
    merged: list[str] = []
    for p in (piece for piece in pieces if piece.strip()):
        if merged and count_tokens(merged[-1]) < min_tokens:
            merged[-1] = merged[-1] + "\n" + p
        else:
            merged.append(p)
    while len(merged) > 1 and count_tokens(merged[-1]) < min_tokens:
        tail = merged.pop()
        merged[-1] = merged[-1] + "\n" + tail
    return merged


class SemanticParentChildChunker:
    file_types: set[str] = {"pdf", "docx"}

    def __init__(self, embeddings: Embeddings | None = None) -> None:
        self._embeddings = embeddings

    def _get_embeddings(self, cfg: IngestionConfig) -> Embeddings:
        emb = self._embeddings
        if emb is None:
            from .embeddings import CachingEmbeddings, build_azure_embeddings

            emb = CachingEmbeddings(build_azure_embeddings(cfg), cfg.cache_dir, enabled=cfg.enable_embedding_cache)
            self._embeddings = emb
        return emb

    def chunk(self, doc: IRDocument, cfg: IngestionConfig) -> list[Chunk]:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        semantic = SemanticChunker(
            self._get_embeddings(cfg),
            breakpoint_threshold_type=cfg.breakpoint_threshold_type,
            breakpoint_threshold_amount=cfg.breakpoint_threshold_amount,
        )
        recursive = RecursiveCharacterTextSplitter(
            chunk_size=cfg.max_child_tokens,
            chunk_overlap=0,
            length_function=count_tokens,
        )

        chunks: list[Chunk] = []
        child_ord = 0
        parent_ord = 0
        root: Chunk | None = None

        def get_root() -> Chunk:
            nonlocal root
            if root is None:
                root = synthetic_root_parent(doc)
                chunks.append(root)
            return root

        for section in doc.sections:
            paragraphs = [tb.text for tb in sorted(section.text_blocks, key=lambda b: b.order)]
            section_text = "\n\n".join(p for p in paragraphs if p.strip())

            table_parent_id: str | None = None

            if section_text.strip():
                parent_texts = _split_by_paragraphs(section_text, cfg.max_parent_tokens)
                n = len(parent_texts)
                for i, ptext in enumerate(parent_texts):
                    spath = list(section.section_path)
                    if n > 1:
                        spath = self._suffix_part(spath, i + 1, n)
                    parent = self._mk_parent(doc, section, spath, ptext, parent_ord)
                    chunks.append(parent)
                    parent_ord += 1
                    if table_parent_id is None:
                        table_parent_id = parent.chunk_id

                    child_ord = self._emit_children(
                        doc, section, parent, semantic, recursive, cfg, chunks, child_ord
                    )

            # tables in this section attach to the section's (first) parent
            if section.tables:
                parent_id = table_parent_id or get_root().chunk_id
                for table in section.tables:
                    new = build_table_chunks(doc, table, parent_id, cfg, child_ord)
                    chunks.extend(new)
                    child_ord += len(new)

        # doc-level orphan tables -> synthetic root
        for table in doc.tables:
            new = build_table_chunks(doc, table, get_root().chunk_id, cfg, child_ord)
            chunks.extend(new)
            child_ord += len(new)

        return chunks

    # ------------------------------------------------------------------ #
    def _emit_children(
        self,
        doc: IRDocument,
        section: Section,
        parent: Chunk,
        semantic: Any,
        recursive: Any,
        cfg: IngestionConfig,
        chunks: list[Chunk],
        child_ord: int,
    ) -> int:
        # tiny section: one child identical to the parent so it stays retrievable
        if count_tokens(parent.text) < cfg.min_child_tokens:
            chunks.append(self._mk_child(doc, parent, parent.text, child_ord))
            return child_ord + 1

        pieces: list[str] = semantic.split_text(parent.text)

        # split oversized pieces first, then merge undersized ones last so the
        # min_child_tokens invariant survives recursive re-splitting (spec 9.8).
        expanded: list[str] = []
        for pc in pieces:
            if count_tokens(pc) > cfg.max_child_tokens:
                expanded.extend(recursive.split_text(pc))
            else:
                expanded.append(pc)
        expanded = _merge_small(expanded, cfg.min_child_tokens)

        for pc in expanded:
            chunks.append(self._mk_child(doc, parent, pc, child_ord))
            child_ord += 1
        return child_ord

    @staticmethod
    def _suffix_part(section_path: list[str], part: int, total: int) -> list[str]:
        spath = list(section_path)
        suffix = f"(part {part}/{total})"
        if spath:
            spath[-1] = f"{spath[-1]} {suffix}"
        else:
            spath = [suffix]
        return spath

    @staticmethod
    def _mk_parent(doc: IRDocument, section: Section, spath: list[str], text: str, ordinal: int) -> Chunk:
        return Chunk(
            chunk_id=compute_chunk_id(doc.doc_id, spath, "parent", ordinal, text),
            doc_id=doc.doc_id,
            parent_id=None,
            role=ChunkRole.PARENT,
            block_type=BlockType.TEXT,
            text=text,
            section_path=spath,
            source_uri=doc.source_uri,
            file_type=doc.file_type,
            page=section.page_start,
            token_count=count_tokens(text),
        )

    @staticmethod
    def _mk_child(doc: IRDocument, parent: Chunk, text: str, ordinal: int) -> Chunk:
        return Chunk(
            chunk_id=compute_chunk_id(doc.doc_id, parent.section_path, "child", ordinal, text),
            doc_id=doc.doc_id,
            parent_id=parent.chunk_id,
            role=ChunkRole.CHILD,
            block_type=BlockType.TEXT,
            text=text,
            section_path=parent.section_path,
            source_uri=doc.source_uri,
            file_type=doc.file_type,
            page=parent.page,
            token_count=count_tokens(text),
        )
