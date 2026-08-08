"""§9.5 / 9.9 + quality gate — pipeline behavior on real files."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_ingestion.config import IngestionConfig
from rag_ingestion.parsers.base import LowExtractionQualityError
from rag_ingestion.parsers.docling_parser import DoclingParser, ParsedElement
from rag_ingestion.pipeline import IngestionPipeline


def _write_csv(path: Path) -> None:
    path.write_text("region,revenue\nNorth,1200\nSouth,3400\n")


def test_ids_deterministic_across_runs(tmp_path: Path, cfg: IngestionConfig) -> None:
    # §9.5 — ingesting the same file twice yields identical chunk_id sets.
    csv = tmp_path / "sales.csv"
    _write_csv(csv)
    pipe = IngestionPipeline(cfg)
    r1 = pipe.ingest_file(csv)
    r2 = pipe.ingest_file(csv)
    assert r1.status in ("success", "partial")
    assert {c.chunk_id for c in r1.chunks} == {c.chunk_id for c in r2.chunks}
    assert r1.doc_id == r2.doc_id


def test_batch_resilience(tmp_path: Path, cfg: IngestionConfig) -> None:
    # §9.9 — a corrupt file fails in isolation; siblings still succeed.
    good = tmp_path / "good.csv"
    _write_csv(good)
    corrupt = tmp_path / "corrupt.xlsx"
    corrupt.write_bytes(b"this is not a real xlsx file")

    results = IngestionPipeline(cfg).ingest_directory(tmp_path)
    by_name = {Path(r.source_uri).name: r for r in results}
    assert by_name["good.csv"].status in ("success", "partial")
    assert by_name["corrupt.xlsx"].status == "failed"
    assert by_name["corrupt.xlsx"].error is not None
    assert by_name["good.csv"].chunks  # siblings still produced output


def test_quality_gate_raises_on_low_extraction(fixtures_dir: Path, cfg: IngestionConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    # Scanned/image-only PDF path: too few chars/page -> LowExtractionQualityError.
    parser = DoclingParser(cfg)

    def fake_extract(_path: Path) -> tuple[list[ParsedElement], int, int]:
        return ([ParsedElement(kind="text", text="x", page=1)], 3, 3)  # 1 char / 3 pages

    monkeypatch.setattr(parser, "_extract", fake_extract)
    with pytest.raises(LowExtractionQualityError):
        parser.parse(fixtures_dir / "scanned.pdf")


def test_pipeline_reports_failed_on_quality_gate(fixtures_dir: Path, cfg: IngestionConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = IngestionPipeline(cfg)

    def fake_extract(_path: Path) -> tuple[list[ParsedElement], int, int]:
        return ([ParsedElement(kind="text", text="x", page=1)], 2, 2)

    # patch the DoclingParser instance the registry resolves for .pdf
    parser = pipe.registry.for_path(fixtures_dir / "scanned.pdf")
    monkeypatch.setattr(parser, "_extract", fake_extract)
    result = pipe.ingest_file(fixtures_dir / "scanned.pdf")
    assert result.status == "failed"
    assert "quality" in (result.error or "").lower() or "page" in (result.error or "").lower()
