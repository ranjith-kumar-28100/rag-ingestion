"""Tier-3 chat LLM provider dispatch (Ollama default / Azure optional).

No network: ChatOllama/AzureChatOpenAI construct lazily and don't connect until
invoked, so these assert wiring only.
"""

from __future__ import annotations

import pytest

from rag_ingestion.config import IngestionConfig
from rag_ingestion.parsers.docling_parser import DoclingParser


def test_default_llm_provider_is_ollama() -> None:
    assert IngestionConfig().llm_provider == "ollama"
    assert IngestionConfig().ollama_model == "mistral"


def test_build_llm_ollama() -> None:
    parser = DoclingParser(IngestionConfig())
    llm = parser._build_llm()
    assert llm.__class__.__name__ == "ChatOllama"


def test_build_llm_azure_requires_credentials() -> None:
    cfg = IngestionConfig(llm_provider="azure")
    with pytest.raises(ValueError, match="azure"):
        DoclingParser(cfg)._build_llm()


def test_build_llm_unknown_provider_raises() -> None:
    cfg = IngestionConfig(llm_provider="bogus")
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        DoclingParser(cfg)._build_llm()
