"""Adapters from real API responses / usage objects to water estimates.

These read actual token usage, so they are the accurate path — prefer them
over prompt-text estimation whenever you have a response in hand.

Supported shapes (duck-typed, dicts or objects):
- OpenAI Chat Completions: usage.prompt_tokens / completion_tokens,
  completion_tokens_details.reasoning_tokens, prompt_tokens_details.cached_tokens
- OpenAI Responses API & Anthropic Messages: usage.input_tokens /
  output_tokens, cache_read_input_tokens (Anthropic),
  output_tokens_details.reasoning_tokens (OpenAI Responses)
"""

from __future__ import annotations

from typing import Any

from soif.estimator import WaterEstimate, estimate


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _int(value: Any) -> int:
    return int(value) if value else 0


def from_usage(usage: Any, model: str | None = None, **kwargs: Any) -> WaterEstimate:
    """Estimate water from a usage object/dict of any major SDK shape.

    Extra keyword arguments (provider=, region=, include_embodied=, ...) are
    forwarded to :func:`soif.estimate`.
    """
    input_tokens = _int(_get(usage, "input_tokens") or _get(usage, "prompt_tokens"))
    output_tokens = _int(_get(usage, "output_tokens") or _get(usage, "completion_tokens"))

    reasoning = _int(
        _get(_get(usage, "completion_tokens_details"), "reasoning_tokens")
        or _get(_get(usage, "output_tokens_details"), "reasoning_tokens")
    )
    # Reasoning tokens are already included in output/completion counts for
    # both OpenAI shapes, so don't double-count energy: keep them within
    # output_tokens and report the split informationally.
    cached = _int(
        _get(usage, "cache_read_input_tokens")
        or _get(_get(usage, "prompt_tokens_details"), "cached_tokens")
    )
    # Cached tokens are included in input counts (OpenAI) or separate
    # (Anthropic cache_read_input_tokens). Normalise: input = uncached part.
    if _get(usage, "cache_read_input_tokens") is None:
        input_tokens = max(0, input_tokens - cached)

    # Anthropic cache writes are ordinary prefill work.
    input_tokens += _int(_get(usage, "cache_creation_input_tokens"))

    est = estimate(
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached,
        **kwargs,
    )
    if reasoning:
        est = _with(est, reasoning_tokens=reasoning, output_tokens=output_tokens - reasoning)
    return est


def from_response(response: Any, model: str | None = None, **kwargs: Any) -> WaterEstimate:
    """Estimate water from a full SDK response (OpenAI or Anthropic).

    Reads ``response.model`` and ``response.usage``; pass ``model=`` to
    override.
    """
    usage = _get(response, "usage")
    if usage is None:
        raise ValueError("response has no .usage; pass token counts to soif.estimate() instead")
    return from_usage(usage, model=model or _get(response, "model"), **kwargs)


def _with(est: WaterEstimate, **changes: Any) -> WaterEstimate:
    from dataclasses import replace

    return replace(est, **changes)
