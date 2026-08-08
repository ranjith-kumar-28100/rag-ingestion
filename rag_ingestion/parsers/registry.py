"""Extension -> parser resolution (spec section 4).

Adding a new format = register a parser here. Nothing downstream changes.
"""

from __future__ import annotations

from pathlib import Path

from ..config import IngestionConfig
from .base import DocumentParser
from .docling_parser import DoclingParser
from .tabular_parser import TabularParser


class ParserRegistry:
    def __init__(self, config: IngestionConfig) -> None:
        self._parsers: list[DocumentParser] = [
            DoclingParser(config),
            TabularParser(),
        ]

    def register(self, parser: DocumentParser) -> None:
        self._parsers.insert(0, parser)

    def for_path(self, path: str | Path) -> DocumentParser:
        ext = Path(path).suffix.lower()
        for parser in self._parsers:
            if ext in parser.supported_extensions:
                return parser
        raise ValueError(f"No parser registered for extension {ext!r}")

    @property
    def supported_extensions(self) -> set[str]:
        exts: set[str] = set()
        for parser in self._parsers:
            exts |= parser.supported_extensions
        return exts
