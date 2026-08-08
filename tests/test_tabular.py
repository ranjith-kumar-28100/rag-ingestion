"""§9.3 / 9.7 — row chunking (headers preserved) and the scale guard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rag_ingestion.chunkers.row import RowChunker
from rag_ingestion.config import IngestionConfig
from rag_ingestion.models import BlockType, ChunkRole
from rag_ingestion.parsers.tabular_parser import TabularParser


def test_row_chunks_contain_column_headers(tmp_path: Path, cfg: IngestionConfig) -> None:
    csv = tmp_path / "sales.csv"
    csv.write_text("region,revenue\nNorth,1200\nSouth,3400\nEast,900\n")
    doc = TabularParser().parse(csv)
    chunks = RowChunker().chunk(doc, cfg)
    row_children = [c for c in chunks if c.role is ChunkRole.CHILD and c.block_type is BlockType.ROW]
    assert row_children
    for c in row_children:
        assert "region" in c.text and "revenue" in c.text
        assert c.text.startswith("Sheet: sales")


def test_headerless_csv_synthesizes_columns(tmp_path: Path) -> None:
    csv = tmp_path / "nums.csv"
    csv.write_text("1,2,3\n4,5,6\n7,8,9\n")
    doc = TabularParser().parse(csv)
    meta = doc.raw_metadata["sheets"]["nums"]
    assert meta["has_header"] is False
    assert meta["columns"] == ["col_0", "col_1", "col_2"]


def test_scale_guard_fires(tmp_path: Path, cfg: IngestionConfig) -> None:
    # §9.7 — 6000-row sheet: only a parent chunk, routing == structured_query.
    xlsx = tmp_path / "big.xlsx"
    pd.DataFrame({"id": range(6000), "v": range(6000)}).to_excel(xlsx, sheet_name="big", index=False)
    doc = TabularParser().parse(xlsx)
    chunks = RowChunker().chunk(doc, cfg)
    assert len(chunks) == 1
    parent = chunks[0]
    assert parent.role is ChunkRole.PARENT
    assert parent.metadata["routing"] == "structured_query"
    assert parent.metadata["row_count"] == 6000


def test_multi_sheet_produces_parent_per_sheet(tmp_path: Path, cfg: IngestionConfig) -> None:
    xlsx = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({"a": [1, 2]}).to_excel(w, sheet_name="one", index=False)
        pd.DataFrame({"b": [3, 4]}).to_excel(w, sheet_name="two", index=False)
    doc = TabularParser().parse(xlsx)
    chunks = RowChunker().chunk(doc, cfg)
    parents = [c for c in chunks if c.role is ChunkRole.PARENT]
    assert {p.sheet_name for p in parents} == {"one", "two"}


def test_row_batching_respects_caps(tmp_path: Path, cfg: IngestionConfig) -> None:
    tuned = cfg.model_copy(update={"max_rows_per_chunk": 3, "row_batch_target_tokens": 100000})
    csv = tmp_path / "rows.csv"
    rows = "\n".join(f"{i},val{i}" for i in range(10))
    csv.write_text("id,name\n" + rows + "\n")
    doc = TabularParser().parse(csv)
    chunks = RowChunker().chunk(doc, tuned)
    row_children = [c for c in chunks if c.role is ChunkRole.CHILD]
    for c in row_children:
        assert c.row_range is not None
        start, end = c.row_range
        assert (end - start + 1) <= 3
