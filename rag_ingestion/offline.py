"""Air-gapped / offline enforcement.

When enabled, libraries are told never to reach the network: HuggingFace and
Transformers switch to offline mode (so a missing model *fails loudly* instead of
being silently downloaded), telemetry is disabled, and LangChain tracing is
forced off. This does not transmit any document data by itself — it removes the
remaining first-run download/telemetry connections so the module runs fully
inside your system.

None of the default processing paths (sentence-transformers embeddings, Ollama
LLM, Docling parsing) send document content off the machine; the Azure providers
are opt-in and are the only paths that would.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("rag_ingestion.offline")

# key -> value forced on. setdefault-style so an explicit env override wins.
_OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",          # huggingface_hub: no hub calls
    "TRANSFORMERS_OFFLINE": "1",    # transformers: no hub calls
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "LANGCHAIN_TRACING_V2": "false",  # no LangSmith tracing
    "LANGCHAIN_TRACING": "false",
    "ANONYMIZED_TELEMETRY": "False",  # various libs' analytics opt-out
}


def enforce_offline() -> None:
    """Force offline/no-telemetry env for local model libraries.

    Must run before the first import of transformers/sentence-transformers, which
    happens lazily at parse/chunk time — so calling this in the pipeline
    constructor is early enough. For the strongest guarantee, also export these in
    the process environment before launching Python.
    """
    applied: list[str] = []
    for key, value in _OFFLINE_ENV.items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    if applied:
        logger.info("offline mode: set %s", ", ".join(applied))
