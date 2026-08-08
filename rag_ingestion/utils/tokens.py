"""Token counting via tiktoken.

A single cached encoder is shared across the process. We use ``cl100k_base``
(the encoding behind text-embedding-3-* and gpt-4o) so counts reflect what the
embedding model actually sees.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Return the number of tokens in ``text``."""
    if not text:
        return 0
    return len(_encoder().encode(text))
