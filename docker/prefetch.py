"""One-time model prefetch — run ONLINE to populate the `model-cache` volume.

After this, api/worker run fully offline (HF_HUB_OFFLINE=1) against the cache.
This is the only step that touches the network, and it downloads only public
model/vocab files — never any document content.

    docker compose --profile prefetch run --rm prefetch
"""

from __future__ import annotations

import subprocess
import sys

from rag_ingestion import IngestionConfig


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


def _prefetch_docling() -> None:
    print("[prefetch] docling models")
    try:
        from docling.utils.model_downloader import download_models

        download_models()
        print("[prefetch] docling models cached (python api)")
        return
    except Exception as exc:  # noqa: BLE001 - fall back to the CLI
        print(f"[prefetch] python api unavailable ({exc}); trying CLI")
    try:
        subprocess.run(["docling-tools", "models", "download"], check=True)
        print("[prefetch] docling models cached (cli)")
    except Exception as exc:  # noqa: BLE001
        print(f"[prefetch] WARNING: docling model download failed: {exc}", file=sys.stderr)


def main() -> None:
    cfg = IngestionConfig()
    _prefetch_embeddings(cfg)
    _prefetch_tokenizer(cfg)
    _prefetch_docling()
    print("[prefetch] done")


if __name__ == "__main__":
    main()
