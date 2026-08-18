# API reference

Everything is importable from the top-level `soif` package. All quantities are
`(low, mid, high)` scenario triples (`soif.Triple`); millilitres for water, Wh for
energy.

## `soif.estimate(...)`

```python
soif.estimate(
    model=None,              # "gpt-4o", "claude-sonnet-4-5", ... (registry lookup)
    prompt=None,             # prompt text; tokens approximated if given
    input_tokens=0,
    output_tokens=None,      # defaults to a typical 500 with an assumption note
    reasoning_tokens=0,      # thinking tokens (charged at output rate)
    cached_tokens=0,         # cache-read tokens (charged at 1%)
    reasoning_effort=None,   # "none" | "low" | "medium" | "high" preset
    tier=None,               # "nano" | "small" | "medium" | "large" | "frontier"
    active_params_b=None,    # derive tier from active parameter count
    provider=None,           # "aws" | "azure" | "gcp" | "average" (WUE/PUE preset)
    region=None,             # "world" | "us" | "eu" | "france" | "nordics" | "asia" | "renewable"
    wue=None, pue=None, ewif=None,   # raw factor overrides (float or Triple)
    include_embodied=True,   # False = operational water only
) -> WaterEstimate
```

Unknown models fall back to the `large` tier and say so in `assumptions` — soif never
guesses silently.

## `WaterEstimate`

| Attribute | Meaning |
|---|---|
| `total_ml` | total water, `Triple` in mL |
| `onsite_ml` / `offsite_ml` / `embodied_ml` | breakdown: cooling / electricity generation / manufacturing |
| `energy_it_wh` / `energy_facility_wh` | server energy and at-the-meter energy |
| `input_tokens`, `output_tokens`, `reasoning_tokens`, `cached_tokens` | token accounting |
| `model`, `tier`, `provider`, `region`, `calls` | resolution details |
| `assumptions` | every default the estimate leaned on |
| `factors_version` | version of the factor set used |

Methods: `humanize()` (one-line summary), `to_dict()` (JSON-ready), and `a + b`
(estimates are additive across calls/nodes).

## Adapters

```python
soif.from_response(response, model=None, **estimate_kwargs)  # OpenAI / Anthropic response
soif.from_usage(usage, model=None, **estimate_kwargs)        # a usage object or dict
```

Handles OpenAI Chat Completions (`prompt_tokens`/`completion_tokens`, reasoning and
cached-token details), OpenAI Responses, and Anthropic Messages (`input_tokens`/
`output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`) without
double-counting.

## `soif.Meter`

```python
meter = soif.Meter(budget_ml=50)     # budget optional
meter.record(est)                    # returns est (pass-through)
meter.total                          # merged WaterEstimate or None
meter.total_ml                       # mid-scenario running total
meter.over_budget                    # bool (advisory; never raises)
meter.remaining_ml
meter.summary()                      # one-line status
```

## `soif.optimize`

```python
optimize.rank(candidates, input_tokens=1000, output_tokens=500, min_tier=None, **kw)
optimize.pick_model(candidates, **kw)      # least-thirsty candidate meeting min_tier
optimize.savings(baseline, alternative, **kw)
# -> {"baseline_ml", "alternative_ml", "saved_ml", "saved_pct"}
```

## Utilities

```python
soif.approx_tokens(text, model=None)   # tiktoken if installed, else chars/4
soif.Triple(low, mid, high)            # supports +, *, to_dict()
soif.factors                           # all factor tables + FACTORS_VERSION
soif.SoifError                         # raised on invalid tier/provider/region/effort
```
