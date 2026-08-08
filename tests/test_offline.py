"""Air-gap / offline enforcement and the no-Azure-when-offline guardrail."""

from __future__ import annotations

import pytest

from rag_ingestion.config import IngestionConfig
from rag_ingestion.offline import _OFFLINE_ENV, enforce_offline
from rag_ingestion.pipeline import IngestionPipeline


def test_offline_defaults_off() -> None:
    assert IngestionConfig().offline is False


def test_offline_with_azure_embedding_is_rejected() -> None:
    with pytest.raises(ValueError, match="off-machine"):
        IngestionConfig(offline=True, embedding_provider="azure")


def test_offline_with_azure_llm_is_rejected() -> None:
    with pytest.raises(ValueError, match="off-machine"):
        IngestionConfig(offline=True, llm_provider="azure")


def test_offline_with_local_providers_ok() -> None:
    cfg = IngestionConfig(offline=True)  # defaults are local
    assert cfg.offline is True


def test_enforce_offline_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    enforce_offline()
    assert os_env_has_offline()


def test_pipeline_offline_enables_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    IngestionPipeline(IngestionConfig(offline=True))
    assert os_env_has_offline()


def os_env_has_offline() -> bool:
    import os

    return os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
