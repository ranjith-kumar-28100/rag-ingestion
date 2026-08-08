"""Chunker layer contract (spec section 6).

Chunkers consume the IR uniformly. They never branch on how a section was
produced (heading/page/LLM/slide) — that all lives in the parser.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import IngestionConfig
from ..models import Chunk, IRDocument


@runtime_checkable
class Chunker(Protocol):
    file_types: set[str]

    def chunk(self, doc: IRDocument, cfg: IngestionConfig) -> list[Chunk]: ...
