"""Intermediate Representation (IR) and output Chunk models.

This is the module contract (spec section 3). Every parser MUST emit an
``IRDocument``; every chunker MUST emit ``list[Chunk]``. Format-specific logic
never travels past the IR.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.documents import Document


class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    SLIDE = "slide"
    ROW = "row"


class ChunkRole(str, Enum):
    PARENT = "parent"
    CHILD = "child"


# --------------------------------------------------------------------------- #
# Intermediate Representation
# --------------------------------------------------------------------------- #
class TableBlock(BaseModel):
    table_id: str
    markdown: str  # canonical serialization for embedding
    rows: list[list[str]]  # raw grid, headers included
    caption: str | None = None
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)


class TextBlock(BaseModel):
    block_id: str
    text: str
    page: int | None = None
    order: int  # document reading order


class Section(BaseModel):
    section_id: str
    heading: str | None  # None => synthetic/unstructured section
    level: int  # 1 = H1; 0 = document root / flat doc
    section_path: list[str]  # e.g. ["3. Architecture", "3.2 Retrieval"]
    text_blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


class IRDocument(BaseModel):
    doc_id: str
    source_uri: str
    file_type: str  # pdf | docx | pptx | xlsx | csv
    title: str | None = None
    sections: list[Section] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)  # doc-level orphan tables
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    parent_id: str | None  # None for PARENT chunks
    role: ChunkRole
    block_type: BlockType
    text: str
    section_path: list[str] = Field(default_factory=list)
    source_uri: str
    file_type: str
    page: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    row_range: tuple[int, int] | None = None
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_langchain(self) -> Document:
        """Emit a LangChain ``Document``.

        ``page_content`` is the chunk text; every other field is flattened into
        ``metadata`` as a scalar (str/int/float/bool). Azure AI Search rejects
        nested objects in filterable fields, so ``section_path`` is joined with
        ``" > "`` and ``row_range`` is split into two ints. Any nested values in
        the free-form ``metadata`` dict are coerced to strings.
        """
        from langchain_core.documents import Document

        meta: dict[str, str | int | float | bool] = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "role": self.role.value,
            "block_type": self.block_type.value,
            "section_path": " > ".join(self.section_path),
            "source_uri": self.source_uri,
            "file_type": self.file_type,
            "token_count": self.token_count,
        }
        if self.parent_id is not None:
            meta["parent_id"] = self.parent_id
        if self.page is not None:
            meta["page"] = self.page
        if self.slide_number is not None:
            meta["slide_number"] = self.slide_number
        if self.sheet_name is not None:
            meta["sheet_name"] = self.sheet_name
        if self.row_range is not None:
            meta["row_start"] = self.row_range[0]
            meta["row_end"] = self.row_range[1]

        for key, value in self.metadata.items():
            meta[key] = _flatten_scalar(value)

        return Document(page_content=self.text, metadata=meta)


def _flatten_scalar(value: Any) -> str | int | float | bool:
    """Coerce an arbitrary metadata value into a flat filterable scalar."""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return " > ".join(str(v) for v in value)
    return str(value)
