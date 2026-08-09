"""One-time model prefetch — run ONLINE to populate the `model-cache` volume.

After this, api/worker run fully offline (HF_HUB_OFFLINE=1) against the cache.
This is the only step that touches the network, and it downloads only public
model/vocab files — never any document content.

    docker compose --profile prefetch run --rm --build prefetch

The critical step is `_prefetch_pdf_pipeline`: it runs the REAL Docling PDF
pipeline (via the actual parser) on a tiny bundled PDF. That downloads exactly
the model repos AND revisions the runtime will request — the layout model is
pinned to a specific revision, and `docling.utils.download_models()` alone
fetched a different one, so offline PDF parsing failed with LocalEntryNotFound.
Driving the parser guarantees an exact match, and also pulls the EasyOCR models
(OCR is enabled) into HOME, which the compose file points at the cache volume.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rag_ingestion import IngestionConfig

SAMPLE_PDF = Path(__file__).with_name("sample_prefetch.pdf")


def _prefetch_embeddings(cfg: IngestionConfig) -> None:
    print(f"[prefetch] embedding model: {cfg.local_embedding_model}")
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(cfg.local_embedding_model, device="cpu")
    print("[prefetch] embedding model cached")


def _prefetch_tokenizer(cfg: IngestionConfig) -> None:
    model = cfg.token_encoder_model or cfg.local_embedding_model
    print(f"[prefetch] tokenizer: {model}")
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(model)
    print("[prefetch] tokenizer cached")


def _prefetch_pdf_pipeline(cfg: IngestionConfig) -> None:
    """Warm the exact runtime PDF pipeline (layout + TableFormer + OCR models)."""
    print("[prefetch] docling PDF pipeline (layout + tableformer + ocr)")
    if not SAMPLE_PDF.exists():
        print(f"[prefetch] WARNING: {SAMPLE_PDF} missing; skipping PDF warm-up", file=sys.stderr)
        return
    from rag_ingestion.parsers.docling_parser import DoclingParser

    parser = DoclingParser(cfg)
    # _extract runs the real Docling conversion, downloading every model/revision
    # the runtime path touches into the HF cache (and EasyOCR models into HOME).
    parser._extract(SAMPLE_PDF)
    print("[prefetch] docling PDF pipeline cached")


def main() -> None:
    cfg = IngestionConfig()
    _prefetch_embeddings(cfg)
    _prefetch_tokenizer(cfg)
    _prefetch_pdf_pipeline(cfg)
    print("[prefetch] done")


if __name__ == "__main__":
    main()
