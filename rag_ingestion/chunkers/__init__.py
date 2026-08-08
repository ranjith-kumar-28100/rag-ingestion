"""Chunker layer: IR -> list[Chunk], dispatched by file_type in the pipeline."""

from .base import Chunker
from .row import RowChunker
from .semantic_parent_child import SemanticParentChildChunker
from .slide import SlideChunker
from .table import TableChunker, build_table_chunks, synthetic_root_parent

__all__ = [
    "Chunker",
    "RowChunker",
    "SemanticParentChildChunker",
    "SlideChunker",
    "TableChunker",
    "build_table_chunks",
    "synthetic_root_parent",
]
