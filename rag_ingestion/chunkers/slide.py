"""PPTX chunker — slide children with a parent window (spec section 6.2).

Child = one chunk per slide (title + body + notes). Parent = a ±slide_window
context window (or a section-divider group when the deck has detectable
section-header slides), deduplicated by content hash. Oversized slides sub-split
semantically but keep the same window parent. Empty (image-only) slides are
skipped and counted into ``doc.raw_metadata["empty_slides"]`` for pipeline stats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import IngestionConfig
from ..models import BlockType, Chunk, ChunkRole, IRDocument, Section
from ..utils.ids import compute_chunk_id, content_hash
from ..utils.tokens import count_tokens

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def _slide_text(section: Section) -> str:
    parts: list[str] = []
    if section.heading:
        parts.append(section.heading)
    parts.extend(tb.text for tb in sorted(section.text_blocks, key=lambda b: b.order) if tb.text.strip())
    return "\n".join(parts)


def _is_empty_slide(section: Section) -> bool:
    return not section.heading and not any(tb.text.strip() for tb in section.text_blocks)


def _is_divider(section: Section) -> bool:
    """Section-header slide heuristic: a title with no meaningful body."""
    return bool(section.heading) and not any(tb.text.strip() for tb in section.text_blocks)


class SlideChunker:
    file_types: set[str] = {"pptx"}

    def __init__(self, embeddings: Embeddings | None = None) -> None:
        self._embeddings = embeddings

    def chunk(self, doc: IRDocument, cfg: IngestionConfig) -> list[Chunk]:
        from ..utils.tokens import configure_token_counter

        configure_token_counter(cfg)
        empty = 0
        slides: list[tuple[Section, str]] = []
        for section in doc.sections:
            if _is_empty_slide(section):
                empty += 1
                continue
            slides.append((section, _slide_text(section)))
        doc.raw_metadata["empty_slides"] = empty

        if not slides:
            return []

        use_sections = any(_is_divider(sec) for sec, _ in slides)
        chunks: list[Chunk] = []
        parent_by_hash: dict[str, Chunk] = {}
        parent_ord = 0
        child_ord = 0

        def get_parent(text: str) -> Chunk:
            nonlocal parent_ord
            h = content_hash(text)
            existing = parent_by_hash.get(h)
            if existing is not None:
                return existing
            parent = self._mk_parent(doc, text, parent_ord)
            parent_by_hash[h] = parent
            chunks.append(parent)
            parent_ord += 1
            return parent

        # resolve, per slide, the parent window text
        parent_texts = self._parent_windows(slides, cfg) if not use_sections else self._section_windows(slides)

        for idx, (section, ctext) in enumerate(slides):
            parent = get_parent(parent_texts[idx])
            for piece in self._child_pieces(ctext, cfg):
                chunks.append(self._mk_child(doc, section, parent, piece, child_ord))
                child_ord += 1

        return chunks

    # ------------------------------------------------------------------ #
    def _parent_windows(self, slides: list[tuple[Section, str]], cfg: IngestionConfig) -> list[str]:
        w = cfg.slide_window
        texts = [t for _, t in slides]
        out: list[str] = []
        for i in range(len(slides)):
            lo, hi = max(0, i - w), min(len(slides), i + w + 1)
            out.append("\n\n".join(texts[lo:hi]))
        return out

    def _section_windows(self, slides: list[tuple[Section, str]]) -> list[str]:
        """Group slides under the most recent divider; parent = the group's text."""
        group_of: list[int] = []
        current = 0
        for i, (sec, _) in enumerate(slides):
            if _is_divider(sec) and i != 0:
                current = i
            group_of.append(current)
        # build group text = concatenation of all slides sharing the group start
        group_text: dict[int, str] = {}
        for start in set(group_of):
            members = [t for j, (_, t) in enumerate(slides) if group_of[j] == start]
            group_text[start] = "\n\n".join(members)
        return [group_text[g] for g in group_of]

    def _child_pieces(self, text: str, cfg: IngestionConfig) -> list[str]:
        if count_tokens(text) <= cfg.max_child_tokens:
            return [text]
        # oversized slide: sub-split semantically
        from langchain_experimental.text_splitter import SemanticChunker

        emb = self._embeddings
        if emb is None:
            from .embeddings import CachingEmbeddings, build_embeddings

            emb = CachingEmbeddings(build_embeddings(cfg), cfg.cache_dir, enabled=cfg.enable_embedding_cache)
            self._embeddings = emb
        semantic = SemanticChunker(
            emb,
            breakpoint_threshold_type=cfg.breakpoint_threshold_type,
            breakpoint_threshold_amount=cfg.breakpoint_threshold_amount,
        )
        pieces: list[str] = semantic.split_text(text)
        return pieces or [text]

    @staticmethod
    def _mk_parent(doc: IRDocument, text: str, ordinal: int) -> Chunk:
        return Chunk(
            chunk_id=compute_chunk_id(doc.doc_id, ["slide-window"], "parent", ordinal, text),
            doc_id=doc.doc_id,
            parent_id=None,
            role=ChunkRole.PARENT,
            block_type=BlockType.SLIDE,
            text=text,
            section_path=["slide-window"],
            source_uri=doc.source_uri,
            file_type=doc.file_type,
            token_count=count_tokens(text),
        )

    @staticmethod
    def _mk_child(doc: IRDocument, section: Section, parent: Chunk, text: str, ordinal: int) -> Chunk:
        return Chunk(
            chunk_id=compute_chunk_id(doc.doc_id, section.section_path, "child", ordinal, text),
            doc_id=doc.doc_id,
            parent_id=parent.chunk_id,
            role=ChunkRole.CHILD,
            block_type=BlockType.SLIDE,
            text=text,
            section_path=section.section_path,
            source_uri=doc.source_uri,
            file_type=doc.file_type,
            slide_number=section.page_start,
            token_count=count_tokens(text),
        )
