"""Ingestion configuration (spec section 8).

All tunable thresholds live here. Chunkers and parsers must read from this
object — no hardcoded thresholds anywhere downstream.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_INGEST_",
        env_file=".env",
        extra="ignore",
    )

    # embeddings (semantic chunker only) --------------------------------------
    # Optional so the tabular/pptx paths and unit tests can run without Azure
    # credentials. The semantic chunker validates presence lazily when it needs
    # to embed.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    embedding_deployment: str = "text-embedding-3-large"
    embedding_batch_size: int = 64

    # semantic chunking -------------------------------------------------------
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: float = 95.0

    # sizing ------------------------------------------------------------------
    max_parent_tokens: int = 3000
    max_child_tokens: int = 800
    min_child_tokens: int = 100

    # tabular -----------------------------------------------------------------
    row_batch_target_tokens: int = 400
    max_rows_per_chunk: int = 10
    max_rows_for_row_chunking: int = 5000

    # pptx --------------------------------------------------------------------
    slide_window: int = 1

    # parent strategy ladder (§5.1.1) -----------------------------------------
    min_headings_for_structure: int = 2
    min_heading_page_coverage: float = 0.6
    enable_llm_toc_inference: bool = False
    toc_inference_deployment: str = "gpt-4o-mini"
    max_heading_candidates: int = 500
    max_heading_chars: int = 120

    # quality gates -----------------------------------------------------------
    min_chars_per_page: int = 50

    # runtime -----------------------------------------------------------------
    max_workers: int = 4
    enable_embedding_cache: bool = True
    cache_dir: Path = Path(".ingest_cache")
