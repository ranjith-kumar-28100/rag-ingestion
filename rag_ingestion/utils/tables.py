"""Grid -> markdown serialization and row-group splitting for tables.

The markdown form is the canonical serialization tables get embedded as. Row
splitting repeats the header row in every fragment (spec section 6.3) so each
fragment is independently interpretable.
"""

from __future__ import annotations

from .tokens import count_tokens


def _escape_cell(value: str) -> str:
    # Pipes would break markdown table columns; newlines collapse a cell.
    return value.replace("|", "\\|").replace("\n", " ").strip()


def grid_to_markdown(rows: list[list[str]], *, has_header: bool = True) -> str:
    """Serialize a 2-D string grid to a GitHub-flavored markdown table.

    The first row is treated as the header when ``has_header`` is true; otherwise
    a synthetic ``col_0..col_n`` header is emitted so the table is still valid
    markdown.
    """
    if not rows:
        return ""

    width = max(len(r) for r in rows)
    norm = [[_escape_cell(c) for c in r] + [""] * (width - len(r)) for r in rows]

    if has_header:
        header, body = norm[0], norm[1:]
    else:
        header = [f"col_{i}" for i in range(width)]
        body = norm

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


def split_table_by_rows(
    rows: list[list[str]],
    max_tokens: int,
    *,
    has_header: bool = True,
) -> list[str]:
    """Split a table into markdown fragments each <= ``max_tokens``.

    The header row is repeated at the top of every fragment. Never splits
    mid-row. If a single data row plus the header already exceeds the budget it
    is still emitted whole (rows are atomic).
    """
    if not rows:
        return []

    header_rows = rows[:1] if has_header else []
    data_rows = rows[1:] if has_header else rows

    if not data_rows:
        return [grid_to_markdown(rows, has_header=has_header)]

    fragments: list[str] = []
    current: list[list[str]] = []

    def flush() -> None:
        if current:
            fragments.append(grid_to_markdown(header_rows + current, has_header=has_header))
            current.clear()

    for row in data_rows:
        candidate = grid_to_markdown(header_rows + current + [row], has_header=has_header)
        if current and count_tokens(candidate) > max_tokens:
            flush()
        current.append(row)

    flush()
    return fragments
