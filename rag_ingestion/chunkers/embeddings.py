"""Embedding client + content-hash cache for the semantic chunker.

The semantic chunker is the only embedding-heavy path in the module. Embeddings
are cached to disk keyed by content hash (spec section 6.1) so re-ingesting an
unchanged document never re-pays. Only ``role == CHILD`` text is ever embedded,
and only during chunking — this module never writes to a vector store.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from langchain_core.embeddings import Embeddings

from ..config import IngestionConfig
from ..utils.ids import content_hash

logger = logging.getLogger("rag_ingestion.chunkers.embeddings")


def build_azure_embeddings(cfg: IngestionConfig) -> Embeddings:
    """Construct the AzureOpenAI embeddings client (lazy import)."""
    from langchain_openai import AzureOpenAIEmbeddings

    if not cfg.azure_openai_endpoint or cfg.azure_openai_api_key is None:
        raise ValueError(
            "Semantic chunking requires azure_openai_endpoint and azure_openai_api_key. "
            "Set RAG_INGEST_AZURE_OPENAI_ENDPOINT / RAG_INGEST_AZURE_OPENAI_API_KEY."
        )
    embeddings: Embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_api_key,
        azure_deployment=cfg.embedding_deployment,
        chunk_size=cfg.embedding_batch_size,
    )
    return embeddings


class CachingEmbeddings(Embeddings):
    """Wrap an ``Embeddings`` with a persistent content-hash cache.

    Subclasses the LangChain ``Embeddings`` interface so it can be handed straight
    to ``SemanticChunker``.
    """

    def __init__(self, base: Embeddings, cache_dir: Path, *, enabled: bool = True) -> None:
        self._base = base
        self._enabled = enabled
        self._path = Path(cache_dir) / "embeddings.json"
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = {}
        if enabled:
            self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._cache = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError) as exc:  # corrupt cache is non-fatal
                logger.warning("embedding cache unreadable, starting fresh: %s", exc)
                self._cache = {}

    def _flush(self) -> None:
        if not self._enabled:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._cache))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._enabled:
            return self._base.embed_documents(texts)

        with self._lock:
            missing = [t for t in texts if content_hash(t) not in self._cache]
            if missing:
                # batch only the cache misses
                fresh = self._base.embed_documents(missing)
                for text, vec in zip(missing, fresh, strict=True):
                    self._cache[content_hash(text)] = vec
                self._flush()
            return [self._cache[content_hash(t)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
