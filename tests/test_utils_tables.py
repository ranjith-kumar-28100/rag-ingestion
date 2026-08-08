"""Direct coverage of grid->markdown and header-repeating row splits (§9.4)."""

from __future__ import annotations

from rag_ingestion.utils.tables import grid_to_markdown, split_table_by_rows


def test_grid_to_markdown_escapes_pipes() -> None:
    md = grid_to_markdown([["a|b", "c"], ["1", "2"]])
    assert "a\\|b" in md
    assert md.count("\n") == 2  # header, separator, one data row


def test_synthetic_header_when_headerless() -> None:
    md = grid_to_markdown([["1", "2"], ["3", "4"]], has_header=False)
    assert "col_0" in md and "col_1" in md


def test_split_repeats_header_in_every_fragment() -> None:
    rows = [["ID", "Name"]] + [[str(i), f"n{i}"] for i in range(60)]
    fragments = split_table_by_rows(rows, max_tokens=25, has_header=True)
    assert len(fragments) > 1
    for frag in fragments:
        assert "ID" in frag and "Name" in frag
