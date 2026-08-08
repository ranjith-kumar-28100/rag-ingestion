"""Ingest one file of each format and print a stats table.

Usage:
    python examples/ingest_demo.py [FIXTURES_DIR]

Embeddings are mocked so the demo runs offline. Docling parsing (pdf/docx/pptx)
downloads models on first run; tabular formats (xlsx/csv) run immediately. Point
it at any directory of sample files, or let it default to tests/fixtures.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from rag_ingestion import IngestionConfig, IngestionPipeline
from rag_ingestion.pipeline import IngestionResult


class _DemoEmbeddings:
    """Offline stand-in for AzureOpenAIEmbeddings so the demo needs no keys."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _print_table(results: list[IngestionResult]) -> None:
    header = f"{'file':<24}{'type':<6}{'status':<9}{'parents':>8}{'children':>9}{'tokens':>8}  {'strategy':<12}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x.stats.file_type):
        name = Path(r.source_uri).name
        s = r.stats
        print(
            f"{name:<24}{s.file_type:<6}{r.status:<9}{s.parent_count:>8}{s.child_count:>9}"
            f"{s.total_tokens:>8}  {(s.parent_strategy or '-'):<12}"
        )
        for w in s.warnings:
            print(f"    ! {w}")


def main() -> None:
    fixtures = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "tests" / "fixtures"
    if not fixtures.exists():
        from tests.generate_fixtures import generate_all  # type: ignore[import-not-found]

        generate_all(fixtures)

    config = IngestionConfig(enable_embedding_cache=False)
    pipeline = IngestionPipeline(config, embeddings=_DemoEmbeddings())

    results = pipeline.ingest_directory(fixtures)
    _print_table(results)


if __name__ == "__main__":
    main()
