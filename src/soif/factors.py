"""Versioned physical factors used by the estimator.

All values are (low, mid, high) scenario triples. They are estimates
assembled from public literature, not measurements. Sources and reasoning
are documented in METHODOLOGY.md; the short version:

- Energy per token by model tier: calibrated against Epoch AI's ~0.3 Wh per
  typical GPT-4o query (~500 output tokens), Google's measured 0.24 Wh per
  median Gemini Apps prompt (full-stack, incl. idle machines and PUE), and
  Mistral's Large 2 lifecycle analysis.
- WUE (on-site Water Usage Effectiveness, L per kWh of IT energy): public
  provider disclosures (AWS 0.15, Microsoft ~0.49, Google fleet ~1.1,
  industry average ~1.8).
- PUE (Power Usage Effectiveness): provider disclosures (Google 1.10,
  hyperscaler ~1.1-1.2, industry average ~1.56).
- EWIF (off-site Electricity Water Intensity Factor, L consumed per kWh
  generated): Macknick et al. / Ren et al. "Making AI Less Thirsty";
  varies strongly with the grid mix.

1 L/kWh is exactly 1 mL/Wh, which keeps the arithmetic tidy.

FACTORS_VERSION bumps whenever any number here changes, so downstream
results can be reproduced against a specific factor set.
"""

from __future__ import annotations

from soif._triple import Triple

FACTORS_VERSION = "2026.08"

# ---------------------------------------------------------------------------
# Model tiers: server-side (IT) energy in Wh per 1000 OUTPUT tokens.
# Tier is chosen from *active* parameters (MoE models count activated
# experts only: DeepSeek-V3 is 671B total but ~37B active -> "medium").
# ---------------------------------------------------------------------------

TIER_WH_PER_1K_OUTPUT_TOKENS: dict[str, Triple] = {
    # < 3B active params (flash-lite / nano class)
    "nano": Triple(0.02, 0.06, 0.15),
    # 3 - 15B active params (mini / flash / haiku class)
    "small": Triple(0.06, 0.20, 0.50),
    # 15 - 70B active params (mid-size, MoE flagships with small active sets)
    "medium": Triple(0.15, 0.50, 1.20),
    # 70 - 250B active params (gpt-4o / sonnet / large dense class)
    "large": Triple(0.30, 0.80, 2.00),
    # > 250B active params, premium frontier & heavy reasoning class
    "frontier": Triple(0.80, 2.00, 5.00),
}

TIER_ORDER = ["nano", "small", "medium", "large", "frontier"]

# Prefill (input) tokens are batched and compute-efficient; per-token energy
# is roughly an order of magnitude below decode.
INPUT_TOKEN_FACTOR = 0.1
# Cached (previously processed) input tokens skip prefill almost entirely.
CACHED_TOKEN_FACTOR = 0.01

# Fallback output length when only a prompt is known (Epoch AI's "typical
# query" assumption).
DEFAULT_OUTPUT_TOKENS = 500

# Extra reasoning ("thinking") tokens generated per expected output token
# when the caller only knows the requested effort level, not actual usage.
REASONING_EFFORT_TOKENS_PER_OUTPUT: dict[str, float] = {
    "none": 0.0,
    "low": 1.0,
    "medium": 4.0,
    "high": 10.0,
}


def tier_from_params(active_params_b: float) -> str:
    if active_params_b < 3:
        return "nano"
    if active_params_b < 15:
        return "small"
    if active_params_b < 70:
        return "medium"
    if active_params_b < 250:
        return "large"
    return "frontier"


# ---------------------------------------------------------------------------
# Data-center profiles: on-site WUE (L/kWh of IT energy) and PUE.
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict[str, Triple]] = {
    "aws": {"wue": Triple(0.10, 0.18, 0.40), "pue": Triple(1.10, 1.15, 1.25)},
    "azure": {"wue": Triple(0.30, 0.49, 0.80), "pue": Triple(1.12, 1.18, 1.30)},
    "gcp": {"wue": Triple(0.90, 1.10, 1.40), "pue": Triple(1.09, 1.10, 1.15)},
    # Non-hyperscaler / unknown colocation.
    "average": {"wue": Triple(0.30, 1.00, 1.90), "pue": Triple(1.20, 1.40, 1.60)},
}

DEFAULT_PROVIDER = "average"

# ---------------------------------------------------------------------------
# Grid regions: EWIF, litres of freshwater *consumed* (not withdrawn) per
# kWh of electricity generated, consumption-weighted for the regional mix.
# ---------------------------------------------------------------------------

REGIONS: dict[str, Triple] = {
    "world": Triple(0.40, 1.50, 3.20),
    "us": Triple(0.50, 1.60, 3.10),
    "eu": Triple(0.30, 1.20, 2.50),
    "france": Triple(0.50, 1.50, 3.00),
    "nordics": Triple(0.05, 0.30, 0.90),
    "asia": Triple(0.50, 1.90, 3.50),
    # 100% matched wind/solar PPAs (lifecycle water of renewables is tiny).
    "renewable": Triple(0.001, 0.10, 0.50),
}

DEFAULT_REGION = "world"

# ---------------------------------------------------------------------------
# Embodied water (chip fabrication, server manufacturing, data-center
# construction) amortised over the useful life, expressed as a multiplier
# on operational water. Literature is thin; the range is deliberately wide
# (Mistral's LCA implies the high end).
# ---------------------------------------------------------------------------

LIFECYCLE_MULTIPLIER = Triple(1.10, 1.50, 3.00)
