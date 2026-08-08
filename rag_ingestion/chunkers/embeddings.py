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


class SentenceTransformerEmbeddings(Embeddings):
    """Fully local embeddings via sentence-transformers — no external calls.

    The model is loaded lazily on first use and reused. Vectors are L2-normalized
    so cosine/percentile breakpoints in SemanticChunker behave consistently.
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 64) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_local_embeddings(cfg: IngestionConfig) -> Embeddings:
    """Construct the local sentence-transformers embeddings."""
    return SentenceTransformerEmbeddings(
        cfg.local_embedding_model,
        device=cfg.local_embedding_device,
        batch_size=cfg.embedding_batch_size,
    )


def build_azure_embeddings(cfg: IngestionConfig) -> Embeddings:
    """Construct the AzureOpenAI embeddings client (lazy import)."""
    from langchain_openai import AzureOpenAIEmbeddings

    if not cfg.azure_openai_endpoint or cfg.azure_openai_api_key is None:
        raise ValueError(
            "Azure embeddings require azure_openai_endpoint and azure_openai_api_key. "
            "Set RAG_INGEST_AZURE_OPENAI_ENDPOINT / RAG_INGEST_AZURE_OPENAI_API_KEY, "
            "or use RAG_INGEST_EMBEDDING_PROVIDER=sentence_transformers for local embeddings."
        )
    embeddings: Embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_api_key,
        azure_deployment=cfg.embedding_deployment,
        chunk_size=cfg.embedding_batch_size,
    )
    return embeddings


def build_embeddings(cfg: IngestionConfig) -> Embeddings:
    """Provider dispatcher: local sentence-transformers or AzureOpenAI."""
    provider = cfg.embedding_provider.lower()
    if provider in ("sentence_transformers", "local", "st"):
        return build_local_embeddings(cfg)
    if provider == "azure":
        return build_azure_embeddings(cfg)
    raise ValueError(
        f"Unknown embedding_provider {cfg.embedding_provider!r}; "
        "expected 'sentence_transformers' or 'azure'."
    )


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
