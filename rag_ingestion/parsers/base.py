"""Parser layer contract (spec section 5).

The ``DocumentParser`` Protocol is the seam. All format-specific logic lives
behind an implementation of it, and the reserved Azure Document Intelligence
fallback parser would slot in here without touching anything downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..models import IRDocument


class ParserError(Exception):
    """Base class for parser failures."""


class LowExtractionQualityError(ParserError):
    """Raised when a document extracts too little text to be usable.

    Triggered by the post-parse quality gate (extracted chars / page count <
    ``min_chars_per_page``). This is the seam where a future Azure Document
    Intelligence fallback parser would take over; we never silently emit an
    empty document.
    """


@runtime_checkable
class DocumentParser(Protocol):
    supported_extensions: set[str]

    def parse(self, path: str | Path) -> IRDocument: ...
