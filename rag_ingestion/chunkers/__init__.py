"""Chunker layer: IR -> list[Chunk], dispatched by file_type in the pipeline."""

from .base import Chunker
from .embeddings import (
    CachingEmbeddings,
    SentenceTransformerEmbeddings,
    build_azure_embeddings,
    build_embeddings,
    build_local_embeddings,
)
from .row import RowChunker
from .semantic_parent_child import SemanticParentChildChunker
from .slide import SlideChunker
from .table import TableChunker, build_table_chunks, synthetic_root_parent

__all__ = [
    "Chunker",
    "CachingEmbeddings",
    "SentenceTransformerEmbeddings",
    "build_embeddings",
    "build_local_embeddings",
    "build_azure_embeddings",
    "RowChunker",
    "SemanticParentChildChunker",
    "SlideChunker",
    "TableChunker",
    "build_table_chunks",
    "synthetic_root_parent",
]
