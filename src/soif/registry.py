"""Model registry: maps model-name substrings to a size tier and a default
hosting profile (provider + region).

Matching is longest-substring-wins on a normalised name, so "gpt-4o-mini"
matches the "gpt-4o-mini" entry rather than "gpt-4o". Unknown models fall
back to the "large" tier with an explicit assumption recorded on the
estimate — never a silent guess.

Tier assignments follow *active* parameter counts where known or credibly
leaked, and product positioning otherwise. They are estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    match: str  # normalised substring to match against the model name
    tier: str
    provider: str  # key into factors.PROVIDERS
    region: str = "world"  # key into factors.REGIONS
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


# OpenAI serves from Microsoft Azure; Anthropic from AWS (and GCP);
# Google from its own fleet; open-weight models default to "average".
MODELS: list[ModelSpec] = [
    # --- OpenAI ---
    ModelSpec("gpt-5-nano", "nano", "azure"),
    ModelSpec("gpt-5-mini", "small", "azure"),
    ModelSpec("gpt-5", "frontier", "azure"),
    ModelSpec("gpt-4.1-nano", "nano", "azure"),
    ModelSpec("gpt-4.1-mini", "small", "azure"),
    ModelSpec("gpt-4.1", "large", "azure"),
    ModelSpec("gpt-4o-mini", "small", "azure"),
    ModelSpec("gpt-4o", "large", "azure"),
    ModelSpec("o4-mini", "medium", "azure"),
    ModelSpec("o3-mini", "medium", "azure"),
    ModelSpec("o3", "frontier", "azure"),
    ModelSpec("o1-mini", "medium", "azure"),
    ModelSpec("o1", "frontier", "azure"),
    # --- Anthropic ---
    ModelSpec("claude-fable", "frontier", "aws"),
    ModelSpec("claude-mythos", "frontier", "aws"),
    ModelSpec("claude-opus", "frontier", "aws"),
    ModelSpec("claude-sonnet", "large", "aws"),
    ModelSpec("claude-haiku", "medium", "aws"),
    ModelSpec("claude-3-5-sonnet", "large", "aws"),
    ModelSpec("claude-3-5-haiku", "medium", "aws"),
    # --- Google ---
    ModelSpec("gemini-2.5-flash-lite", "nano", "gcp"),
    ModelSpec("gemini-2.5-flash", "small", "gcp"),
    ModelSpec("gemini-2.5-pro", "large", "gcp"),
    ModelSpec("gemini-2.0-flash", "small", "gcp"),
    ModelSpec("gemini-1.5-flash", "small", "gcp"),
    ModelSpec("gemini-1.5-pro", "large", "gcp"),
    ModelSpec("gemma-2", "small", "average"),
    ModelSpec("gemma-3", "small", "average"),
    # --- Meta (open weights, tier by dense params) ---
    ModelSpec("llama-3.1-405b", "frontier", "average"),
    ModelSpec("llama-3.1-70b", "large", "average"),
    ModelSpec("llama-3.3-70b", "large", "average"),
    ModelSpec("llama-3.1-8b", "small", "average"),
    ModelSpec("llama-3.2-3b", "small", "average"),
    ModelSpec("llama-3.2-1b", "nano", "average"),
    ModelSpec("llama-4-maverick", "medium", "average", notes="MoE, ~17B active"),
    ModelSpec("llama-4-scout", "medium", "average", notes="MoE, ~17B active"),
    # --- Mistral ---
    ModelSpec("mistral-large", "large", "average", region="france"),
    ModelSpec("mistral-medium", "medium", "average", region="france"),
    ModelSpec("mistral-small", "medium", "average", region="france"),
    ModelSpec("ministral", "small", "average", region="france"),
    ModelSpec("mixtral-8x22b", "medium", "average", notes="MoE, ~39B active"),
    ModelSpec("mixtral-8x7b", "small", "average", notes="MoE, ~13B active"),
    # --- DeepSeek (MoE: 671B total, ~37B active) ---
    ModelSpec("deepseek-v3", "medium", "average", region="asia"),
    ModelSpec("deepseek-r1", "medium", "average", region="asia", notes="count reasoning tokens"),
    # --- xAI ---
    ModelSpec("grok-4", "frontier", "average", region="us"),
    ModelSpec("grok-3", "frontier", "average", region="us"),
    # --- Qwen ---
    ModelSpec("qwen3-235b", "medium", "average", region="asia", notes="MoE, ~22B active"),
    ModelSpec("qwen3-32b", "medium", "average", region="asia"),
    ModelSpec("qwen3-8b", "small", "average", region="asia"),
]

FALLBACK_TIER = "large"


def _normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-").replace(".", "-")


def resolve(model_name: str) -> ModelSpec | None:
    """Return the best-matching spec for a model name, or None.

    Longest matching substring wins; date suffixes, provider prefixes
    ("openai/gpt-4o", "us.anthropic.claude-sonnet-4-5...") and separator
    variations are tolerated because matching is substring-based on a
    normalised form.
    """
    name = _normalise(model_name)
    best: ModelSpec | None = None
    for spec in MODELS:
        if _normalise(spec.match) in name and (best is None or len(spec.match) > len(best.match)):
            best = spec
    return best


def known_models() -> list[ModelSpec]:
    return list(MODELS)
