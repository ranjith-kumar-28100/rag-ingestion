"""XLSX / CSV chunker — batched row chunks with header preservation (spec 6.4).

Each sheet yields one summary PARENT and a series of ROW children. Rows are
packed into batches (bounded by ``row_batch_target_tokens`` and
``max_rows_per_chunk``) to avoid one-row-per-chunk index bloat. The per-row line
already carries every column key (built by the tabular parser), so headers are
preserved in every chunk. Sheets over ``max_rows_for_row_chunking`` are not
row-chunked — only the parent is emitted, flagged for structured-query routing.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import IngestionConfig
from ..models import BlockType, Chunk, ChunkRole, IRDocument, Section
from ..utils.ids import compute_chunk_id
from ..utils.tokens import count_tokens

logger = logging.getLogger("rag_ingestion.chunkers.row")


class RowChunker:
    file_types: set[str] = {"xlsx", "csv"}

    def chunk(self, doc: IRDocument, cfg: IngestionConfig) -> list[Chunk]:
        from ..utils.tokens import configure_token_counter

        configure_token_counter(cfg)
        sheets_meta: dict[str, Any] = doc.raw_metadata.get("sheets", {})
        chunks: list[Chunk] = []
        parent_ord = 0
        child_ord = 0

        for section in doc.sections:
            sheet_name = section.heading or section.section_path[0] if section.section_path else "sheet"
            meta = sheets_meta.get(sheet_name, {})
            row_count = int(meta.get("row_count", len(section.text_blocks)))
            sparse = bool(meta.get("sparse_sheet", False))

            over_scale = row_count > cfg.max_rows_for_row_chunking

            parent = self._mk_parent(doc, section, sheet_name, meta, parent_ord, sparse, over_scale)
            chunks.append(parent)
            parent_ord += 1

            if over_scale:
                logger.warning(
                    "scale guard: sheet %r has %d rows (> %d); emitting parent only, routing=structured_query",
                    sheet_name, row_count, cfg.max_rows_for_row_chunking,
                )
                continue

            for batch in self._batch_rows(section, cfg):
                chunks.append(self._mk_row_chunk(doc, section, parent, sheet_name, batch, cfg, child_ord, sparse))
                child_ord += 1

        return chunks

    # ------------------------------------------------------------------ #
    def _batch_rows(self, section: Section, cfg: IngestionConfig) -> list[list[tuple[int, str]]]:
        blocks = sorted(section.text_blocks, key=lambda b: b.order)
        batches: list[list[tuple[int, str]]] = []
        current: list[tuple[int, str]] = []
        for tb in blocks:
            current.append((tb.order, tb.text))
            body = "\n".join(t for _, t in current)
            over_tokens = count_tokens(body) >= cfg.row_batch_target_tokens
            if len(current) >= cfg.max_rows_per_chunk or over_tokens:
                batches.append(current)
                current = []
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _mk_parent(
        doc: IRDocument,
        section: Section,
        sheet_name: str,
        meta: dict[str, Any],
        ordinal: int,
        sparse: bool,
        over_scale: bool,
    ) -> Chunk:
        columns = meta.get("columns", [])
        dtypes = meta.get("dtypes", {})
        row_count = int(meta.get("row_count", len(section.text_blocks)))
        sample = [tb.text for tb in sorted(section.text_blocks, key=lambda b: b.order)[:3]]

        text_lines = [
            f"Sheet: {sheet_name}",
            f"Columns: {', '.join(str(c) for c in columns)}",
            f"Rows: {row_count}",
            "Dtypes: " + ", ".join(f"{c}={dtypes.get(c, 'object')}" for c in columns),
        ]
        if sample:
            text_lines.append("Sample:")
            text_lines.extend(sample)
        text = "\n".join(text_lines)

        metadata: dict[str, Any] = {"row_count": row_count, "column_count": len(columns)}
        if sparse:
            metadata["sparse_sheet"] = True
        if over_scale:
            metadata["routing"] = "structured_query"

        return Chunk(
            chunk_id=compute_chunk_id(doc.doc_id, [sheet_name], "parent", ordinal, text),
            doc_id=doc.doc_id,
            parent_id=None,
            role=ChunkRole.PARENT,
            block_type=BlockType.ROW,
            text=text,
            section_path=[sheet_name],
            source_uri=doc.source_uri,
            file_type=doc.file_type,
            sheet_name=sheet_name,
            token_count=count_tokens(text),
            metadata=metadata,
        )

    @staticmethod
    def _mk_row_chunk(
        doc: IRDocument,
        section: Section,
        parent: Chunk,
        sheet_name: str,
        batch: list[tuple[int, str]],
        cfg: IngestionConfig,
        ordinal: int,
        sparse: bool,
    ) -> Chunk:
        row_lines = [t for _, t in batch]
        text = f"Sheet: {sheet_name}\n" + "\n".join(row_lines)
        start, end = batch[0][0], batch[-1][0]

        metadata: dict[str, Any] = {}
        if sparse:
            metadata["sparse_sheet"] = True

        return Chunk(
            chunk_id=compute_chunk_id(doc.doc_id, [sheet_name], "child", ordinal, text),
            doc_id=doc.doc_id,
            parent_id=parent.chunk_id,
            role=ChunkRole.CHILD,
            block_type=BlockType.ROW,
            text=text,
            section_path=[sheet_name],
            source_uri=doc.source_uri,
            file_type=doc.file_type,
            sheet_name=sheet_name,
            row_range=(start, end),
            token_count=count_tokens(text),
            metadata=metadata,
        )
