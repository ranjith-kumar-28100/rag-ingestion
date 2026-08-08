"""rag_ingestion — parser-agnostic multi-format document ingestion & chunking.

file in -> list[Chunk] out. No embedding, indexing, or retrieval happens here.
"""

from __future__ import annotations

from .config import IngestionConfig
from .offline import enforce_offline
from .models import (
    BlockType,
    Chunk,
    ChunkRole,
    IRDocument,
    Section,
    TableBlock,
    TextBlock,
)
from .parsers.base import LowExtractionQualityError, ParserError
from .pipeline import IngestionPipeline, IngestionResult, IngestionStats

__all__ = [
    "IngestionConfig",
    "IngestionPipeline",
    "IngestionResult",
    "IngestionStats",
    "IRDocument",
    "Section",
    "TextBlock",
    "TableBlock",
    "Chunk",
    "BlockType",
    "ChunkRole",
    "ParserError",
    "LowExtractionQualityError",
    "enforce_offline",
]

__version__ = "0.1.0"
