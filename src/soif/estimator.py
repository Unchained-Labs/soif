"""Core water-footprint estimator.

The model follows the operational-water methodology of Ren et al.,
"Making AI Less Thirsty" (arXiv:2304.03271), with an optional embodied
(lifecycle) adder:

    E_it       = per-token server energy x tokens            [Wh]
    E_facility = E_it x PUE                                  [Wh]
    W_onsite   = E_it x WUE            (cooling evaporation)  [mL]
    W_offsite  = E_facility x EWIF     (power generation)     [mL]
    W_embodied = (W_onsite + W_offsite) x (lifecycle - 1)     [mL]
    W_total    = W_onsite + W_offsite + W_embodied            [mL]

Every factor is a (low, mid, high) scenario triple; see factors.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from soif import factors, registry, tokens
from soif._triple import Triple

_ML_PER_TEASPOON = 4.93
_ML_PER_BOTTLE = 500.0
_ML_PER_DROP = 0.05


class SoifError(ValueError):
    pass


@dataclass(frozen=True)
class WaterEstimate:
    """Water and energy footprint of one or more LLM calls, in millilitres.

    All quantities are (low, mid, high) uncertainty triples. Estimates are
    additive: ``a + b`` merges two estimates (e.g. across agent-graph nodes).
    """

    total_ml: Triple
    onsite_ml: Triple
    offsite_ml: Triple
    embodied_ml: Triple
    energy_it_wh: Triple
    energy_facility_wh: Triple
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    model: str = ""
    tier: str = ""
    provider: str = ""
    region: str = ""
    calls: int = 1
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    factors_version: str = factors.FACTORS_VERSION

    # -- composition --------------------------------------------------------

    def __add__(self, other: WaterEstimate) -> WaterEstimate:
        if not isinstance(other, WaterEstimate):
            return NotImplemented

        def merge(a: str, b: str) -> str:
            if a == b:
                return a
            return ", ".join(x for x in (a, b) if x) or ""

        return WaterEstimate(
            total_ml=self.total_ml + other.total_ml,
            onsite_ml=self.onsite_ml + other.onsite_ml,
            offsite_ml=self.offsite_ml + other.offsite_ml,
            embodied_ml=self.embodied_ml + other.embodied_ml,
            energy_it_wh=self.energy_it_wh + other.energy_it_wh,
            energy_facility_wh=self.energy_facility_wh + other.energy_facility_wh,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            model=merge(self.model, other.model),
            tier=merge(self.tier, other.tier),
            provider=merge(self.provider, other.provider),
            region=merge(self.region, other.region),
            calls=self.calls + other.calls,
            assumptions=tuple(dict.fromkeys(self.assumptions + other.assumptions)),
        )

    # -- presentation -------------------------------------------------------

    def humanize(self) -> str:
        """One-line human summary of the mid estimate with its range."""
        mid = self.total_ml.mid
        if mid < 1.0:
            unit = f"{mid / _ML_PER_DROP:.1f} drops"
        elif mid < 25:
            unit = f"{mid / _ML_PER_TEASPOON:.1f} teaspoons"
        else:
            unit = f"{mid / _ML_PER_BOTTLE:.2f} x 500 mL bottles"
        return (
            f"~{_fmt_ml(mid)} of water ({unit}); "
            f"range {_fmt_ml(self.total_ml.low)} - {_fmt_ml(self.total_ml.high)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "water_ml": {
                "total": self.total_ml.to_dict(),
                "onsite_cooling": self.onsite_ml.to_dict(),
                "offsite_electricity": self.offsite_ml.to_dict(),
                "embodied": self.embodied_ml.to_dict(),
            },
            "energy_wh": {
                "it": self.energy_it_wh.to_dict(),
                "facility": self.energy_facility_wh.to_dict(),
            },
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "reasoning": self.reasoning_tokens,
                "cached": self.cached_tokens,
            },
            "model": self.model,
            "tier": self.tier,
            "provider": self.provider,
            "region": self.region,
            "calls": self.calls,
            "assumptions": list(self.assumptions),
            "factors_version": self.factors_version,
        }


def _fmt_ml(ml: float) -> str:
    if ml >= 1000:
        return f"{ml / 1000:.2f} L"
    if ml >= 1:
        return f"{ml:.2f} mL"
    return f"{ml:.3f} mL"


def estimate(
    model: str | None = None,
    *,
    prompt: str | None = None,
    input_tokens: int = 0,
    output_tokens: int | None = None,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    reasoning_effort: str | None = None,
    tier: str | None = None,
    active_params_b: float | None = None,
    provider: str | None = None,
    region: str | None = None,
    wue: float | Triple | None = None,
    pue: float | Triple | None = None,
    ewif: float | Triple | None = None,
    include_embodied: bool = True,
) -> WaterEstimate:
    """Estimate the water footprint of a single LLM call.

    Provide either real token usage (``input_tokens`` / ``output_tokens`` /
    ``reasoning_tokens``, e.g. from an API response) or a ``prompt`` string,
    in which case tokens are approximated and the output length defaults to
    a typical 500 tokens.

    The model is resolved from the registry by name; unknown models fall
    back to the "large" tier unless ``tier`` or ``active_params_b`` says
    otherwise. ``provider``/``region``/``wue``/``pue``/``ewif`` override the
    hosting profile. Set ``include_embodied=False`` for operational water
    only (the scope of Google's published Gemini number).
    """
    assumptions: list[str] = []

    # -- resolve the model tier and hosting profile -------------------------
    spec = registry.resolve(model) if model else None
    if tier is None and active_params_b is not None:
        tier = factors.tier_from_params(active_params_b)
        assumptions.append(f"tier '{tier}' derived from {active_params_b:g}B active params")
    if tier is None:
        if spec is not None:
            tier = spec.tier
        else:
            tier = registry.FALLBACK_TIER
            label = f"'{model}'" if model else "unspecified model"
            assumptions.append(
                f"unknown model {label}: assumed tier '{tier}' "
                "(pass tier= or active_params_b= to override)"
            )
    if tier not in factors.TIER_WH_PER_1K_OUTPUT_TOKENS:
        raise SoifError(f"unknown tier '{tier}'; expected one of {factors.TIER_ORDER}")

    provider = provider or (spec.provider if spec else factors.DEFAULT_PROVIDER)
    if provider not in factors.PROVIDERS:
        raise SoifError(f"unknown provider '{provider}'; expected one of {list(factors.PROVIDERS)}")
    region = region or (spec.region if spec else factors.DEFAULT_REGION)
    if region not in factors.REGIONS:
        raise SoifError(f"unknown region '{region}'; expected one of {list(factors.REGIONS)}")

    # -- resolve token counts ------------------------------------------------
    if prompt is not None:
        prompt_tokens = tokens.approx_tokens(prompt, model)
        input_tokens += prompt_tokens
        assumptions.append(f"prompt text approximated as {prompt_tokens} input tokens")
    if output_tokens is None:
        output_tokens = factors.DEFAULT_OUTPUT_TOKENS
        assumptions.append(f"assumed a typical {output_tokens}-token response")
    if reasoning_effort is not None:
        if reasoning_effort not in factors.REASONING_EFFORT_TOKENS_PER_OUTPUT:
            raise SoifError(
                f"unknown reasoning_effort '{reasoning_effort}'; expected one of "
                f"{list(factors.REASONING_EFFORT_TOKENS_PER_OUTPUT)}"
            )
        extra = round(output_tokens * factors.REASONING_EFFORT_TOKENS_PER_OUTPUT[reasoning_effort])
        if extra:
            reasoning_tokens += extra
            assumptions.append(
                f"reasoning effort '{reasoning_effort}' modelled as {extra} thinking tokens"
            )

    # -- energy --------------------------------------------------------------
    e_per_1k = factors.TIER_WH_PER_1K_OUTPUT_TOKENS[tier]
    effective_output = output_tokens + reasoning_tokens
    effective_input = (
        input_tokens * factors.INPUT_TOKEN_FACTOR
        + cached_tokens * factors.CACHED_TOKEN_FACTOR
    )
    energy_it_wh = e_per_1k * ((effective_output + effective_input) / 1000.0)

    pue_t = Triple.of(pue) if pue is not None else factors.PROVIDERS[provider]["pue"]
    energy_facility_wh = energy_it_wh * pue_t

    # -- water ---------------------------------------------------------------
    wue_t = Triple.of(wue) if wue is not None else factors.PROVIDERS[provider]["wue"]
    ewif_t = Triple.of(ewif) if ewif is not None else factors.REGIONS[region]

    onsite_ml = energy_it_wh * wue_t  # 1 L/kWh == 1 mL/Wh
    offsite_ml = energy_facility_wh * ewif_t
    operational_ml = onsite_ml + offsite_ml
    if include_embodied:
        embodied_ml = operational_ml * (factors.LIFECYCLE_MULTIPLIER + (-1.0))
    else:
        embodied_ml = Triple.zero()
        assumptions.append("embodied (manufacturing) water excluded")
    total_ml = operational_ml + embodied_ml

    return WaterEstimate(
        total_ml=total_ml,
        onsite_ml=onsite_ml,
        offsite_ml=offsite_ml,
        embodied_ml=embodied_ml,
        energy_it_wh=energy_it_wh,
        energy_facility_wh=energy_facility_wh,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        model=model or "",
        tier=tier,
        provider=provider,
        region=region,
        assumptions=tuple(assumptions),
    )
