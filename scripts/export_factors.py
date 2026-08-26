#!/usr/bin/env python3
"""Emit ``factors.json``: the machine-readable form of the factor tables.

The numbers in ``soif.factors`` and ``soif.registry`` are the single source of
truth for the whole soif project, but they live in Python and other languages
need them too (``soif-web`` recomputes estimates in TypeScript). Hand-porting
the tables guarantees drift, so this script serialises them instead.

Two things make the port checkable rather than merely convenient:

* the JSON carries the *structural* inputs an implementation needs — tiers,
  token weights, tier boundaries, provider and region tables, the model
  registry — not just the headline factors;
* it carries ``parity_vectors``: estimates computed here, by the reference
  implementation, that a port must reproduce. A port that matches every vector
  has demonstrated parity rather than asserted it.

Usage::

    python scripts/export_factors.py            # write factors.json
    python scripts/export_factors.py --check    # fail if factors.json is stale
    python scripts/export_factors.py --stdout   # print, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import soif  # noqa: E402
from soif import factors, registry  # noqa: E402
from soif._triple import Triple  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "factors.json"

# Bump when the *shape* of this document changes, independently of
# FACTORS_VERSION which tracks the numbers. Consumers should refuse a schema
# major they do not understand.
SCHEMA_VERSION = "1.0"

# Rounded to a fixed number of decimals so the file is byte-stable across
# platforms and Python versions; 12 is far beyond the precision any factor
# claims while staying well inside float64's exact-repr range.
FLOAT_DECIMALS = 12


def _num(value: float) -> float:
    """Round to a stable decimal count and normalise -0.0 to 0.0."""
    rounded = round(float(value), FLOAT_DECIMALS)
    return rounded + 0.0


def _triple(triple: Triple) -> dict[str, float]:
    return {"low": _num(triple.low), "mid": _num(triple.mid), "high": _num(triple.high)}


def _tier_boundaries() -> list[dict[str, Any]]:
    """Reconstruct ``factors.tier_from_params`` as data.

    Kept as an explicit table rather than probed numerically so that a change
    to the function without a change here is caught by ``test_factors_export``.
    """
    return [
        {"tier": "nano", "max_active_params_b": 3},
        {"tier": "small", "max_active_params_b": 15},
        {"tier": "medium", "max_active_params_b": 70},
        {"tier": "large", "max_active_params_b": 250},
        {"tier": "frontier", "max_active_params_b": None},
    ]


# Inputs chosen to exercise every branch a port has to get right: cached and
# reasoning tokens (the two weights most easily mis-scaled), each provider and
# region table, the unknown-model fallback, explicit tier and active-params
# overrides, raw wue/pue/ewif overrides, and embodied on/off.
PARITY_CASES: list[dict[str, Any]] = [
    {"id": "gpt-4o-typical", "model": "gpt-4o", "output_tokens": 500},
    {"id": "gpt-4o-with-input", "model": "gpt-4o", "input_tokens": 1200, "output_tokens": 500},
    {
        "id": "cached-heavy-agentic",
        "model": "claude-sonnet-4-5",
        "input_tokens": 2,
        "cached_tokens": 994_232,
        "output_tokens": 333,
    },
    {
        "id": "reasoning-explicit",
        "model": "o3",
        "output_tokens": 500,
        "reasoning_tokens": 8000,
    },
    {
        "id": "reasoning-effort-high",
        "model": "gpt-5",
        "output_tokens": 500,
        "reasoning_effort": "high",
    },
    {"id": "opus-frontier-aws", "model": "claude-opus-4", "output_tokens": 1000},
    {"id": "haiku-medium-aws", "model": "claude-haiku-4-5", "output_tokens": 1000},
    {"id": "gemini-flash-gcp", "model": "gemini-2.5-flash", "output_tokens": 500},
    {"id": "gemini-flash-lite-nano", "model": "gemini-2.5-flash-lite", "output_tokens": 500},
    {"id": "mistral-large-france", "model": "mistral-large", "output_tokens": 500},
    {"id": "deepseek-v3-asia", "model": "deepseek-v3", "output_tokens": 500},
    {"id": "grok-4-us", "model": "grok-4", "output_tokens": 500},
    {"id": "unknown-model-fallback", "model": "totally-made-up-model", "output_tokens": 500},
    {"id": "no-model-at-all", "output_tokens": 500},
    {
        "id": "region-nordics-override",
        "model": "llama-3.1-70b",
        "output_tokens": 500,
        "provider": "aws",
        "region": "nordics",
    },
    {
        "id": "region-renewable-operational-only",
        "model": "gpt-4o",
        "output_tokens": 500,
        "region": "renewable",
        "include_embodied": False,
    },
    {
        "id": "raw-factor-overrides",
        "model": "my-fine-tune",
        "output_tokens": 500,
        "active_params_b": 8,
        "wue": 0.2,
        "pue": 1.12,
        "ewif": 0.4,
    },
    {
        "id": "explicit-tier-nano-azure-eu",
        "output_tokens": 500,
        "tier": "nano",
        "provider": "azure",
        "region": "eu",
    },
    {
        "id": "provider-average-world",
        "output_tokens": 500,
        "tier": "medium",
        "provider": "average",
        "region": "world",
    },
    {
        "id": "zero-output-input-only",
        "model": "gpt-4o",
        "input_tokens": 10_000,
        "output_tokens": 0,
    },
]


def _parity_vector(case: dict[str, Any]) -> dict[str, Any]:
    case_id = case["id"]
    kwargs = {k: v for k, v in case.items() if k != "id"}
    model = kwargs.pop("model", None)
    est = soif.estimate(model, **kwargs)

    inputs: dict[str, Any] = dict(kwargs)
    if model is not None:
        inputs["model"] = model

    return {
        "id": case_id,
        "input": inputs,
        "expected": {
            "tier": est.tier,
            "provider": est.provider,
            "region": est.region,
            "energy_it_wh": _triple(est.energy_it_wh),
            "energy_facility_wh": _triple(est.energy_facility_wh),
            "onsite_ml": _triple(est.onsite_ml),
            "offsite_ml": _triple(est.offsite_ml),
            "embodied_ml": _triple(est.embodied_ml),
            "total_ml": _triple(est.total_ml),
        },
    }


def build_document() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "factors_version": factors.FACTORS_VERSION,
        "soif_version": soif.__version__,
        "source": "https://github.com/Unchained-Labs/soif",
        "generated_by": "scripts/export_factors.py",
        "notice": (
            "Generated from src/soif/factors.py and src/soif/registry.py. "
            "Do not edit by hand — CI fails if this file drifts from the Python. "
            "All triples are (low, mid, high) scenario bounds, not confidence "
            "intervals. See METHODOLOGY.md."
        ),
        "model": {
            "description": (
                "E_it = (output + reasoning + input_token_factor * input "
                "+ cached_token_factor * cached) / 1000 * wh_per_1k_output_tokens[tier]; "
                "E_facility = E_it * pue; W_onsite = E_it * wue; "
                "W_offsite = E_facility * ewif; "
                "W_embodied = (W_onsite + W_offsite) * (lifecycle_multiplier - 1); "
                "W_total = W_onsite + W_offsite + W_embodied."
            ),
            "units": {
                "energy": "Wh",
                "water": "mL",
                "wue": "L/kWh of IT energy (1 L/kWh == 1 mL/Wh)",
                "ewif": "L/kWh of electricity generated",
                "pue": "dimensionless",
            },
        },
        "tiers": {
            "order": list(factors.TIER_ORDER),
            "wh_per_1k_output_tokens": {
                tier: _triple(factors.TIER_WH_PER_1K_OUTPUT_TOKENS[tier])
                for tier in factors.TIER_ORDER
            },
            "boundaries_active_params_b": _tier_boundaries(),
            "fallback_tier": registry.FALLBACK_TIER,
        },
        "token_weights": {
            "input_token_factor": _num(factors.INPUT_TOKEN_FACTOR),
            "cached_token_factor": _num(factors.CACHED_TOKEN_FACTOR),
            "reasoning_token_factor": 1.0,
            "default_output_tokens": factors.DEFAULT_OUTPUT_TOKENS,
            "reasoning_effort_tokens_per_output": {
                effort: _num(value)
                for effort, value in sorted(factors.REASONING_EFFORT_TOKENS_PER_OUTPUT.items())
            },
        },
        "providers": {
            name: {"wue": _triple(spec["wue"]), "pue": _triple(spec["pue"])}
            for name, spec in sorted(factors.PROVIDERS.items())
        },
        "default_provider": factors.DEFAULT_PROVIDER,
        "regions": {name: _triple(t) for name, t in sorted(factors.REGIONS.items())},
        "default_region": factors.DEFAULT_REGION,
        "lifecycle_multiplier": _triple(factors.LIFECYCLE_MULTIPLIER),
        "registry": {
            "normalisation": {
                "lowercase": True,
                "strip": True,
                "replace": {"_": "-", " ": "-", ".": "-"},
            },
            "match_rule": (
                "Normalise the model name and each spec's `match`, keep every spec whose "
                "normalised match is a substring of the normalised name, and take the one "
                "with the longest `match`. No match means fallback_tier with "
                "default_provider and default_region."
            ),
            "models": [
                {
                    "match": spec.match,
                    "tier": spec.tier,
                    "provider": spec.provider,
                    "region": spec.region,
                    **({"notes": spec.notes} if spec.notes else {}),
                    **({"aliases": list(spec.aliases)} if spec.aliases else {}),
                }
                for spec in registry.known_models()
            ],
        },
        "parity_vectors": [_parity_vector(case) for case in PARITY_CASES],
    }


def serialise(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the checked-in factors.json differs from a fresh export",
    )
    parser.add_argument("--stdout", action="store_true", help="write to stdout instead of a file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"output path (default: {OUTPUT_PATH.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    payload = serialise(build_document())

    if args.stdout:
        sys.stdout.write(payload)
        return 0

    if args.check:
        if not args.output.exists():
            print(
                f"{args.output} is missing. Run: python scripts/export_factors.py",
                file=sys.stderr,
            )
            return 1
        current = args.output.read_text(encoding="utf-8")
        if current != payload:
            print(
                f"{args.output.name} is out of date with src/soif/factors.py.\n"
                "Run: python scripts/export_factors.py",
                file=sys.stderr,
            )
            return 1
        print(f"{args.output.name} is up to date (factor set {factors.FACTORS_VERSION}).")
        return 0

    args.output.write_text(payload, encoding="utf-8")
    print(
        f"wrote {args.output.relative_to(REPO_ROOT)} "
        f"(factor set {factors.FACTORS_VERSION}, schema {SCHEMA_VERSION})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
