"""Token counting with a swappable backend.

Two backends:

* **tiktoken** (``cl100k_base``) — matches OpenAI/Azure embedding + chat models.
  Used for the online (Azure) path. Downloads the BPE vocab on first use.
* **HF tokenizer** — a local ``transformers`` tokenizer (defaults to the local
  embedding model's own tokenizer). Used for the local / offline path so counts
  reflect what the local embedder actually sees and nothing is fetched from an
  OpenAI endpoint.

The active backend is process-global and selected via :func:`configure_token_counter`
(called by the pipeline at construction and by each chunker before it runs).
Default, if never configured, is tiktoken.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import IngestionConfig

TokenCounter = Callable[[str], int]

_LOCAL_EMBEDDING_ALIASES = ("sentence_transformers", "local", "st")


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _tiktoken_encoder() -> object:
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _tiktoken_count(text: str) -> int:
    return len(_tiktoken_encoder().encode(text))  # type: ignore[attr-defined]


@lru_cache(maxsize=4)
def _hf_tokenizer(model: str) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def _hf_counter(model: str) -> TokenCounter:
    tokenizer = _hf_tokenizer(model)

    def count(text: str) -> int:
        # .tokenize avoids special tokens and the model-max-length encode warning
        return len(tokenizer.tokenize(text))  # type: ignore[attr-defined]

    return count


# --------------------------------------------------------------------------- #
# active backend
# --------------------------------------------------------------------------- #
_active: TokenCounter | None = None


def set_token_counter(backend: str, hf_model: str | None = None) -> None:
    """Set the process-global token backend: ``"tiktoken"`` or ``"hf"``."""
    global _active
    if backend == "hf":
        if not hf_model:
            raise ValueError("hf token backend requires a model name")
        _active = _hf_counter(hf_model)
    elif backend == "tiktoken":
        _active = _tiktoken_count
    else:
        raise ValueError(f"Unknown token backend {backend!r}; expected 'tiktoken' or 'hf'.")


def configure_token_counter(cfg: IngestionConfig) -> None:
    """Select the backend from config.

    ``token_counter``: ``"auto"`` (default), ``"tiktoken"`` or ``"hf"``. Under
    ``auto``, the HF tokenizer is used when offline or when embeddings are local;
    otherwise tiktoken.
    """
    mode = cfg.token_counter.lower()
    if mode == "auto":
        use_hf = cfg.offline or cfg.embedding_provider.lower() in _LOCAL_EMBEDDING_ALIASES
        mode = "hf" if use_hf else "tiktoken"

    if mode == "hf":
        set_token_counter("hf", cfg.token_encoder_model or cfg.local_embedding_model)
    else:
        set_token_counter("tiktoken")


def count_tokens(text: str) -> int:
    """Return the number of tokens in ``text`` using the active backend."""
    if not text:
        return 0
    if _active is None:
        return _tiktoken_count(text)
    return _active(text)
