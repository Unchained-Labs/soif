"""Water-aware model routing for agent graphs.

The single biggest water lever in a chain of agent prompts is *which model
each node calls* — tiers differ by ~30x per token. These helpers let a
router node pick the least-thirsty candidate that still meets a capability
floor, and quantify what a downgrade saves.
"""

from __future__ import annotations

from dataclasses import dataclass

from soif import factors
from soif.estimator import WaterEstimate, estimate


@dataclass(frozen=True)
class RankedModel:
    model: str
    tier: str
    estimate: WaterEstimate

    @property
    def ml(self) -> float:
        return self.estimate.total_ml.mid


def rank(
    candidates: list[str],
    *,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    min_tier: str | None = None,
    **kwargs: object,
) -> list[RankedModel]:
    """Rank candidate models by mid-scenario water use for a workload.

    ``min_tier`` filters out models below a capability floor (a node that
    needs frontier-grade reasoning shouldn't be routed to a nano model).
    Extra kwargs are forwarded to :func:`soif.estimate`.
    """
    floor = factors.TIER_ORDER.index(min_tier) if min_tier else 0
    ranked = []
    for name in candidates:
        est = estimate(name, input_tokens=input_tokens, output_tokens=output_tokens, **kwargs)
        if factors.TIER_ORDER.index(est.tier) >= floor:
            ranked.append(RankedModel(model=name, tier=est.tier, estimate=est))
    return sorted(ranked, key=lambda r: r.ml)


def pick_model(candidates: list[str], **kwargs: object) -> str:
    """Return the least-thirsty candidate meeting the constraints of rank()."""
    ranked = rank(candidates, **kwargs)
    if not ranked:
        raise ValueError("no candidate model satisfies the constraints")
    return ranked[0].model


def savings(baseline: str, alternative: str, **kwargs: object) -> dict[str, float]:
    """Mid-scenario water saved per call by switching baseline -> alternative."""
    base = estimate(baseline, **kwargs)
    alt = estimate(alternative, **kwargs)
    saved = base.total_ml.mid - alt.total_ml.mid
    return {
        "baseline_ml": base.total_ml.mid,
        "alternative_ml": alt.total_ml.mid,
        "saved_ml": saved,
        "saved_pct": (saved / base.total_ml.mid * 100.0) if base.total_ml.mid else 0.0,
    }
