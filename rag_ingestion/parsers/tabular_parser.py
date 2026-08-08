"""Tabular parser — xlsx, csv (spec section 5.2).

pandas + openpyxl give row-level control Docling does not expose. Each row
becomes a ``TextBlock`` whose text is the serialized ``col: val | ...`` line
(the per-row half of the §6.4 format); the batch-level ``Sheet:`` header and the
sheet summary parent are assembled later by the row chunker. Sheet-level
structural facts (columns, dtypes, header presence, sparsity) are stashed in
``IRDocument.raw_metadata["sheets"]`` for the chunker to build parents and
headers from.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..models import IRDocument, Section, TextBlock
from ..utils.ids import doc_id_for_path

logger = logging.getLogger("rag_ingestion.parsers.tabular")

_HEADER_NUMERIC_RATIO = 0.5  # >50% numeric row-0 cells => headerless
_SPARSE_EMPTY_RATIO = 0.4  # >40% empty cells => pivot/dashboard-like


def _coerce_cell(value: Any) -> str:
    """NaN -> '', strip, coerce to str; render integer-valued floats without '.0'."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return str(value).strip()
    return str(value).strip()


def _is_numeric(text: str) -> bool:
    if not text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _looks_headerless(row0: list[str]) -> bool:
    if not row0:
        return False
    numeric = sum(1 for c in row0 if _is_numeric(c))
    return (numeric / len(row0)) > _HEADER_NUMERIC_RATIO


def _infer_dtype(values: list[str]) -> str:
    non_empty = [v for v in values if v]
    if not non_empty:
        return "empty"
    if all(v.lstrip("-").isdigit() for v in non_empty):
        return "int64"
    if all(_is_numeric(v) for v in non_empty):
        return "float64"
    return "object"


def _serialize_row(columns: list[str], cells: list[str]) -> str:
    """`col_1: val_1 | col_2: val_2 | ...` — keys always present (§6.4)."""
    padded = cells + [""] * (len(columns) - len(cells))
    return " | ".join(f"{col}: {padded[i]}" for i, col in enumerate(columns))


class TabularParser:
    supported_extensions: set[str] = {".xlsx", ".csv"}

    def parse(self, path: str | Path) -> IRDocument:
        p = Path(path)
        ext = p.suffix.lower()
        doc_id = doc_id_for_path(p)

        if ext == ".csv":
            sheets = {p.stem: pd.read_csv(p, header=None, dtype=str, keep_default_na=True)}
            file_type = "csv"
        elif ext == ".xlsx":
            sheets = pd.read_excel(p, sheet_name=None, header=None, dtype=object)
            file_type = "xlsx"
        else:  # pragma: no cover - registry guards this
            raise ValueError(f"TabularParser cannot handle extension {ext!r}")

        doc = IRDocument(
            doc_id=doc_id,
            source_uri=p.resolve().as_uri(),
            file_type=file_type,
            title=p.stem,
        )
        sheet_meta: dict[str, Any] = {}

        for idx, (sheet_name, df) in enumerate(sheets.items()):
            section, meta = self._parse_sheet(doc_id, idx, str(sheet_name), df)
            doc.sections.append(section)
            sheet_meta[str(sheet_name)] = meta

        doc.raw_metadata["sheets"] = sheet_meta
        return doc

    def _parse_sheet(
        self,
        doc_id: str,
        index: int,
        sheet_name: str,
        df: pd.DataFrame,
    ) -> tuple[Section, dict[str, Any]]:
        grid: list[list[str]] = [[_coerce_cell(c) for c in row] for row in df.itertuples(index=False, name=None)]

        section_id = f"{doc_id}-sheet-{index}"
        section = Section(
            section_id=section_id,
            heading=sheet_name,
            level=1,
            section_path=[sheet_name],
        )

        if not grid:
            meta = {"columns": [], "dtypes": {}, "has_header": False, "sparse_sheet": False, "row_count": 0}
            return section, meta

        width = max(len(r) for r in grid)
        grid = [r + [""] * (width - len(r)) for r in grid]

        headerless = _looks_headerless(grid[0])
        if headerless:
            columns = [f"col_{i}" for i in range(width)]
            data_rows = grid
        else:
            columns = [c or f"col_{i}" for i, c in enumerate(grid[0])]
            data_rows = grid[1:]

        # sparsity heuristic over the data cells
        total_cells = len(data_rows) * width if data_rows else 0
        empty_cells = sum(1 for r in data_rows for c in r if not c)
        sparse = total_cells > 0 and (empty_cells / total_cells) > _SPARSE_EMPTY_RATIO
        if sparse:
            logger.warning(
                "sparse_sheet detected",
                extra={"sheet": sheet_name, "empty_ratio": round(empty_cells / total_cells, 3)},
            )

        dtypes = {
            col: _infer_dtype([r[i] for r in data_rows]) for i, col in enumerate(columns)
        }

        for row_idx, cells in enumerate(data_rows):
            section.text_blocks.append(
                TextBlock(
                    block_id=f"{section_id}-r{row_idx}",
                    text=_serialize_row(columns, cells),
                    order=row_idx,
                )
            )

        meta = {
            "columns": columns,
            "dtypes": dtypes,
            "has_header": not headerless,
            "sparse_sheet": sparse,
            "row_count": len(data_rows),
        }
        return section, meta
