"""Token counting.

Uses tiktoken when installed (``pip install soif[tokenizers]``), otherwise a
chars/4 heuristic — accurate enough for a water estimate whose dominant
uncertainty is the energy factors, not the token count.
"""

from __future__ import annotations


def approx_tokens(text: str, model: str | None = None) -> int:
    """Approximate the token count of *text* for *model*."""
    if not text:
        return 0
    try:
        import tiktoken

        try:
            if model:
                enc = tiktoken.encoding_for_model(model)
            else:
                enc = tiktoken.get_encoding("o200k_base")
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except ImportError:
        # ~4 characters per token is a reasonable cross-model average for
        # English prose and code.
        return max(1, round(len(text) / 4))
