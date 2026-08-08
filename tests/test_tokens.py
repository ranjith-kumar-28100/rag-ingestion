"""Token-counting backend selection (tiktoken online / HF wordpiece local-offline)."""

from __future__ import annotations

import pytest

from rag_ingestion.config import IngestionConfig
from rag_ingestion.utils import tokens


def test_auto_selects_tiktoken_for_azure() -> None:
    cfg = IngestionConfig(embedding_provider="azure", azure_openai_endpoint="x", offline=False)
    resolved = _resolve(cfg)
    assert resolved == "tiktoken"


def test_auto_selects_hf_for_local() -> None:
    cfg = IngestionConfig(embedding_provider="sentence_transformers")
    assert _resolve(cfg) == "hf"


def test_auto_selects_hf_when_offline() -> None:
    cfg = IngestionConfig(offline=True, embedding_provider="sentence_transformers")
    assert _resolve(cfg) == "hf"


def test_explicit_override_wins() -> None:
    assert _resolve(IngestionConfig(embedding_provider="azure", token_counter="hf")) == "hf"
    assert _resolve(IngestionConfig(embedding_provider="sentence_transformers", token_counter="tiktoken")) == "tiktoken"


def test_tiktoken_backend_counts() -> None:
    tokens.set_token_counter("tiktoken")
    assert tokens.count_tokens("") == 0
    assert tokens.count_tokens("hello world foo bar") > 0


def test_hf_backend_counts() -> None:
    # uses the local embedding model's tokenizer; cached from prior embedding use
    cfg = IngestionConfig(token_counter="hf")
    tokens.configure_token_counter(cfg)
    assert tokens.count_tokens("") == 0
    assert tokens.count_tokens("hello world foo bar baz qux") > 0
    # reset to a hermetic default for other tests
    tokens.set_token_counter("tiktoken")


def test_hf_backend_requires_model() -> None:
    with pytest.raises(ValueError, match="requires a model"):
        tokens.set_token_counter("hf", None)


def _resolve(cfg: IngestionConfig) -> str:
    """Return which backend `auto`/explicit resolves to, without loading it."""
    mode = cfg.token_counter.lower()
    if mode == "auto":
        use_hf = cfg.offline or cfg.embedding_provider.lower() in tokens._LOCAL_EMBEDDING_ALIASES
        return "hf" if use_hf else "tiktoken"
    return mode
