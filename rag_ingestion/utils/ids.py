"""Deterministic ID generation (spec section 7).

IDs must be reproducible across runs so re-ingestion upserts rather than
duplicates:

    doc_id    = sha256(source_uri + file_mtime)[:16]
    chunk_id  = sha256(doc_id + section_path + role + ordinal + text)[:16]

The same helper (:func:`content_hash`) also keys the embedding cache, so an
unchanged document never re-pays for embeddings.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path

_SEP = "\x1f"  # unit separator — cannot collide with path/text content


def _sha16(parts: Iterable[str]) -> str:
    h = hashlib.sha256(_SEP.join(parts).encode("utf-8"))
    return h.hexdigest()[:16]


def compute_doc_id(source_uri: str, file_mtime: float) -> str:
    """Deterministic document id from source uri + mtime."""
    return _sha16([source_uri, repr(file_mtime)])


def doc_id_for_path(path: str | Path) -> str:
    """Convenience: derive a doc_id from a filesystem path using its mtime."""
    p = Path(path)
    mtime = os.path.getmtime(p)
    return compute_doc_id(p.resolve().as_uri(), mtime)


def compute_chunk_id(
    doc_id: str,
    section_path: list[str],
    role: str,
    ordinal: int,
    text: str,
) -> str:
    """Deterministic chunk id.

    ``ordinal`` disambiguates otherwise-identical chunks (e.g. a child that
    equals its parent, or repeated row batches).
    """
    return _sha16([doc_id, " > ".join(section_path), role, str(ordinal), text])


def content_hash(text: str) -> str:
    """Stable hash of arbitrary content — used as the embedding cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
