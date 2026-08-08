"""Ingestion pipeline — the public entrypoint (spec section 7).

file in -> IngestionResult (list[Chunk] + stats). Dispatch is by
``IRDocument.file_type``, so adding a format is a parser + chunker registration,
never a change here. Per-file failures are isolated: one bad file never aborts a
batch.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from .chunkers.base import Chunker
from .chunkers.row import RowChunker
from .chunkers.semantic_parent_child import SemanticParentChildChunker
from .chunkers.slide import SlideChunker
from .config import IngestionConfig
from .models import Chunk, ChunkRole, IRDocument
from .parsers.registry import ParserRegistry
from .utils.ids import doc_id_for_path

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger("rag_ingestion.pipeline")


class IngestionStats(BaseModel):
    file_type: str
    parse_seconds: float
    chunk_count: int = 0
    parent_count: int = 0
    child_count: int = 0
    counts_by_block_type: dict[str, int] = Field(default_factory=dict)
    total_tokens: int = 0
    parent_strategy: str | None = None
    empty_slides: int | None = None
    warnings: list[str] = Field(default_factory=list)


class IngestionResult(BaseModel):
    doc_id: str
    source_uri: str
    status: Literal["success", "partial", "failed"]
    chunks: list[Chunk] = Field(default_factory=list)
    stats: IngestionStats
    error: str | None = None


class IngestionPipeline:
    def __init__(self, config: IngestionConfig, embeddings: Embeddings | None = None) -> None:
        self.config = config
        if config.offline:
            from .offline import enforce_offline

            enforce_offline()
        self.registry = ParserRegistry(config)

        spc = SemanticParentChildChunker(embeddings)
        slide = SlideChunker(embeddings)
        row = RowChunker()
        self._chunkers: dict[str, Chunker] = {}
        for chunker in (spc, slide, row):
            for ft in chunker.file_types:
                self._chunkers[ft] = chunker

    # ------------------------------------------------------------------ #
    def ingest_file(self, path: str | Path) -> IngestionResult:
        p = Path(path)
        source_uri = p.resolve().as_uri()
        start = time.perf_counter()
        try:
            parser = self.registry.for_path(p)
            doc = parser.parse(p)
            chunker = self._chunkers.get(doc.file_type)
            if chunker is None:
                raise ValueError(f"No chunker registered for file_type {doc.file_type!r}")
            chunks = chunker.chunk(doc, self.config)
            parse_seconds = round(time.perf_counter() - start, 4)

            warnings = _collect_warnings(doc, chunks)
            stats = _build_stats(doc, chunks, parse_seconds, warnings)
            status: Literal["success", "partial"] = "partial" if warnings else "success"

            self._log(doc.doc_id, doc.file_type, parse_seconds, stats, status)
            return IngestionResult(
                doc_id=doc.doc_id,
                source_uri=source_uri,
                status=status,
                chunks=chunks,
                stats=stats,
            )
        except Exception as exc:  # noqa: BLE001 - per-file isolation is required
            parse_seconds = round(time.perf_counter() - start, 4)
            ext = p.suffix.lstrip(".").lower()
            try:
                doc_id = doc_id_for_path(p)
            except OSError:
                doc_id = ""
            logger.error(
                json.dumps(
                    {"event": "ingest_failed", "doc_id": doc_id, "source_uri": source_uri,
                     "file_type": ext, "error": str(exc), "parse_seconds": parse_seconds}
                )
            )
            return IngestionResult(
                doc_id=doc_id,
                source_uri=source_uri,
                status="failed",
                chunks=[],
                stats=IngestionStats(file_type=ext, parse_seconds=parse_seconds, warnings=[str(exc)]),
                error=str(exc),
            )

    def ingest_directory(self, path: str | Path, glob: str = "**/*") -> list[IngestionResult]:
        base = Path(path)
        supported = self.registry.supported_extensions
        files = [
            p for p in sorted(base.glob(glob))
            if p.is_file() and p.suffix.lower() in supported
        ]
        results: list[IngestionResult] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {pool.submit(self.ingest_file, p): p for p in files}
            for fut in as_completed(futures):
                results.append(fut.result())  # ingest_file never raises
        # deterministic ordering for callers/tests
        results.sort(key=lambda r: r.source_uri)
        return results

    # ------------------------------------------------------------------ #
    def _log(
        self, doc_id: str, file_type: str, parse_seconds: float, stats: IngestionStats, status: str
    ) -> None:
        logger.info(
            json.dumps(
                {
                    "event": "ingest_ok",
                    "doc_id": doc_id,
                    "file_type": file_type,
                    "status": status,
                    "parse_seconds": parse_seconds,
                    "chunk_count": stats.chunk_count,
                    "parent_count": stats.parent_count,
                    "child_count": stats.child_count,
                    "parent_strategy": stats.parent_strategy,
                    "warnings": stats.warnings,
                }
            )
        )


def _build_stats(
    doc: IRDocument, chunks: list[Chunk], parse_seconds: float, warnings: list[str]
) -> IngestionStats:
    by_block: Counter[str] = Counter(c.block_type.value for c in chunks)
    parents = sum(1 for c in chunks if c.role is ChunkRole.PARENT)
    children = sum(1 for c in chunks if c.role is ChunkRole.CHILD)
    return IngestionStats(
        file_type=doc.file_type,
        parse_seconds=parse_seconds,
        chunk_count=len(chunks),
        parent_count=parents,
        child_count=children,
        counts_by_block_type=dict(by_block),
        total_tokens=sum(c.token_count for c in chunks),
        parent_strategy=doc.raw_metadata.get("parent_strategy"),
        empty_slides=doc.raw_metadata.get("empty_slides"),
        warnings=warnings,
    )


def _collect_warnings(doc: IRDocument, chunks: list[Chunk]) -> list[str]:
    warnings: list[str] = []
    strategy = doc.raw_metadata.get("parent_strategy")
    if strategy in ("PAGE", "FLAT"):
        warnings.append(f"parent structure degraded to {strategy}")

    if any(c.metadata.get("routing") == "structured_query" for c in chunks):
        warnings.append("scale guard fired: sheet routed to structured_query")

    if any(c.metadata.get("sparse_sheet") for c in chunks):
        warnings.append("sparse sheet detected (pivot/dashboard-like)")

    empty = doc.raw_metadata.get("empty_slides")
    if isinstance(empty, int) and empty > 0:
        warnings.append(f"{empty} empty slide(s) skipped")

    return warnings
