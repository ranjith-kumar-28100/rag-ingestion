"""Ingestion configuration (spec section 8).

All tunable thresholds live here. Chunkers and parsers must read from this
object — no hardcoded thresholds anywhere downstream.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_INGEST_",
        env_file=".env",
        extra="ignore",
    )

    # embeddings (semantic chunker only) --------------------------------------
    # Provider selects where embeddings come from:
    #   "sentence_transformers" -> fully local, no credentials (default)
    #   "azure"                 -> AzureOpenAIEmbeddings
    embedding_provider: str = "sentence_transformers"
    embedding_batch_size: int = 64

    # local (sentence-transformers) — used when provider == sentence_transformers
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_embedding_device: str = "cpu"

    # azure — optional so the local/tabular/pptx paths and unit tests run without
    # credentials. Validated lazily only when the azure provider actually embeds.
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    embedding_deployment: str = "text-embedding-3-large"

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
    max_heading_candidates: int = 500
    max_heading_chars: int = 120

    # tier-3 TOC-inference LLM provider:
    #   "ollama" -> local Ollama (default), e.g. Mistral 7B
    #   "azure"  -> AzureChatOpenAI (uses toc_inference_deployment)
    llm_provider: str = "ollama"
    ollama_model: str = "mistral"
    ollama_base_url: str = "http://localhost:11434"
    toc_inference_deployment: str = "gpt-4o-mini"  # azure only

    # quality gates -----------------------------------------------------------
    min_chars_per_page: int = 50

    # runtime -----------------------------------------------------------------
    max_workers: int = 4
    enable_embedding_cache: bool = True
    cache_dir: Path = Path(".ingest_cache")

    # air-gap: force local libs offline + no telemetry (see rag_ingestion.offline).
    # Models must be pre-cached; missing assets then fail loudly instead of being
    # fetched. Does not affect which provider is used — set the *_provider fields
    # to their local values (the defaults) to keep document data on the machine.
    offline: bool = False

    @model_validator(mode="after")
    def _no_azure_when_offline(self) -> "IngestionConfig":
        """Fail fast if offline is requested but an Azure provider would egress data."""
        if self.offline:
            outbound = [
                name
                for name, value in (
                    ("embedding_provider", self.embedding_provider),
                    ("llm_provider", self.llm_provider),
                )
                if value.lower() == "azure"
            ]
            if outbound:
                raise ValueError(
                    f"offline=True but {', '.join(outbound)} = 'azure' would send document "
                    "data off-machine. Use the local providers (sentence_transformers / ollama)."
                )
        return self
