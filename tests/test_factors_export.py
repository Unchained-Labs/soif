"""Guards on ``factors.json``: the artifact other languages build against.

``soif-web`` recomputes estimates in TypeScript from this file, so a silent
divergence between the JSON and ``src/soif/factors.py`` would make two repos
disagree about the same number while both look healthy. These tests are the
tripwire; the CI job runs ``--check`` for the same reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import soif
from soif import factors, registry

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_factors  # noqa: E402

FACTORS_JSON = REPO_ROOT / "factors.json"


@pytest.fixture(scope="module")
def document() -> dict:
    assert FACTORS_JSON.exists(), "factors.json is missing; run scripts/export_factors.py"
    return json.loads(FACTORS_JSON.read_text(encoding="utf-8"))


def test_checked_in_file_matches_a_fresh_export():
    """The whole point: the JSON is generated, never edited."""
    fresh = export_factors.serialise(export_factors.build_document())
    current = FACTORS_JSON.read_text(encoding="utf-8")
    assert current == fresh, (
        "factors.json is out of date with src/soif/factors.py. "
        "Run: python scripts/export_factors.py"
    )


def test_versions_are_stamped(document):
    assert document["factors_version"] == factors.FACTORS_VERSION
    assert document["soif_version"] == soif.__version__
    assert document["schema_version"] == export_factors.SCHEMA_VERSION


def test_tier_table_is_complete(document):
    exported = document["tiers"]["wh_per_1k_output_tokens"]
    assert set(exported) == set(factors.TIER_WH_PER_1K_OUTPUT_TOKENS)
    assert document["tiers"]["order"] == factors.TIER_ORDER
    assert document["tiers"]["fallback_tier"] == registry.FALLBACK_TIER
    for tier, triple in factors.TIER_WH_PER_1K_OUTPUT_TOKENS.items():
        assert exported[tier] == {"low": triple.low, "mid": triple.mid, "high": triple.high}


def test_tier_boundaries_reproduce_tier_from_params(document):
    """``_tier_boundaries`` is hand-written, so prove it against the function.

    A boundary edited in ``factors.tier_from_params`` without a matching edit
    to the exporter would otherwise ship a JSON that quietly disagrees.
    """
    boundaries = document["tiers"]["boundaries_active_params_b"]
    assert [b["tier"] for b in boundaries] == factors.TIER_ORDER
    assert boundaries[-1]["max_active_params_b"] is None, "the top tier must be unbounded"

    def tier_from_table(active_params_b: float) -> str:
        for entry in boundaries:
            ceiling = entry["max_active_params_b"]
            if ceiling is None or active_params_b < ceiling:
                return entry["tier"]
        raise AssertionError("boundary table has no terminal tier")

    # Probe each declared edge and its neighbourhood, plus the extremes.
    probes = [0.0, 0.5, 1e-9, 1_000.0, 1e6]
    for entry in boundaries:
        ceiling = entry["max_active_params_b"]
        if ceiling is not None:
            probes += [ceiling - 0.01, ceiling, ceiling + 0.01]
    for value in probes:
        assert tier_from_table(value) == factors.tier_from_params(value), (
            f"boundary table and tier_from_params disagree at {value}B active params"
        )


def test_provider_and_region_tables_match(document):
    assert set(document["providers"]) == set(factors.PROVIDERS)
    assert document["default_provider"] == factors.DEFAULT_PROVIDER
    for name, spec in factors.PROVIDERS.items():
        exported = document["providers"][name]
        assert exported["wue"] == spec["wue"].to_dict()
        assert exported["pue"] == spec["pue"].to_dict()

    assert set(document["regions"]) == set(factors.REGIONS)
    assert document["default_region"] == factors.DEFAULT_REGION
    for name, triple in factors.REGIONS.items():
        assert document["regions"][name] == triple.to_dict()

    assert document["lifecycle_multiplier"] == factors.LIFECYCLE_MULTIPLIER.to_dict()


def test_token_weights_match(document):
    weights = document["token_weights"]
    assert weights["input_token_factor"] == factors.INPUT_TOKEN_FACTOR
    assert weights["cached_token_factor"] == factors.CACHED_TOKEN_FACTOR
    assert weights["default_output_tokens"] == factors.DEFAULT_OUTPUT_TOKENS
    assert (
        weights["reasoning_effort_tokens_per_output"]
        == factors.REASONING_EFFORT_TOKENS_PER_OUTPUT
    )
    # Reasoning tokens are charged as output tokens; the estimator hard-codes
    # this by adding them together, so the exported weight must stay 1.0.
    assert weights["reasoning_token_factor"] == 1.0


def test_registry_is_complete(document):
    exported = document["registry"]["models"]
    assert len(exported) == len(registry.known_models())
    by_match = {entry["match"]: entry for entry in exported}
    for spec in registry.known_models():
        entry = by_match[spec.match]
        assert entry["tier"] == spec.tier
        assert entry["provider"] == spec.provider
        assert entry["region"] == spec.region


@pytest.mark.parametrize("index", range(len(export_factors.PARITY_CASES)))
def test_parity_vectors_reproduce(document, index):
    """Every vector must still be what the estimator produces today.

    A port proves parity by matching these; that only means anything if they
    track the reference implementation, so re-derive rather than trust.
    """
    vector = document["parity_vectors"][index]
    inputs = dict(vector["input"])
    model = inputs.pop("model", None)
    estimate = soif.estimate(model, **inputs)

    expected = vector["expected"]
    assert estimate.tier == expected["tier"]
    assert estimate.provider == expected["provider"]
    assert estimate.region == expected["region"]
    for field in (
        "energy_it_wh",
        "energy_facility_wh",
        "onsite_ml",
        "offsite_ml",
        "embodied_ml",
        "total_ml",
    ):
        actual = getattr(estimate, field)
        for bound in ("low", "mid", "high"):
            assert getattr(actual, bound) == pytest.approx(expected[field][bound], rel=1e-9), (
                f"{vector['id']}: {field}.{bound} drifted"
            )


def test_parity_vectors_cover_the_tricky_paths(document):
    """A port that only ever sees plain output tokens has proved very little."""
    vectors = {v["id"]: v for v in document["parity_vectors"]}
    inputs = [v["input"] for v in document["parity_vectors"]]

    assert any(i.get("cached_tokens") for i in inputs), "no cached-token vector"
    assert any(i.get("reasoning_tokens") for i in inputs), "no explicit-reasoning vector"
    assert any(i.get("reasoning_effort") for i in inputs), "no reasoning-effort vector"
    assert any(i.get("include_embodied") is False for i in inputs), "no operational-only vector"
    assert any("wue" in i for i in inputs), "no raw-override vector"
    assert any("model" not in i for i in inputs), "no model-less vector"

    # The unknown-model path must land on the documented fallback, not a guess.
    fallback = vectors["unknown-model-fallback"]["expected"]
    assert fallback["tier"] == registry.FALLBACK_TIER
    assert fallback["provider"] == factors.DEFAULT_PROVIDER
    assert fallback["region"] == factors.DEFAULT_REGION

    # Every tier and every provider table should appear somewhere.
    covered_tiers = {v["expected"]["tier"] for v in document["parity_vectors"]}
    missing_tiers = set(factors.TIER_ORDER) - covered_tiers
    assert not missing_tiers, f"tiers not covered by any parity vector: {sorted(missing_tiers)}"

    covered_providers = {v["expected"]["provider"] for v in document["parity_vectors"]}
    missing_providers = set(factors.PROVIDERS) - covered_providers
    assert not missing_providers, (
        f"providers not covered by any parity vector: {sorted(missing_providers)}"
    )


def test_export_is_deterministic():
    """Byte-stability is what makes the CI drift check meaningful."""
    first = export_factors.serialise(export_factors.build_document())
    second = export_factors.serialise(export_factors.build_document())
    assert first == second
