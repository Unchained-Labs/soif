# soif 💧

> *soif* — French for **thirst**.

**Estimate the water footprint of LLM prompts, the way you estimate their cost.**

Every LLM answer evaporates real freshwater: data-center cooling towers (on-site) and the
power plants feeding them (off-site) both consume it. Published per-prompt figures span two
orders of magnitude — Google measured **0.26 mL** per median Gemini prompt, while Mistral's
lifecycle analysis reports **45 mL** per 400-token Large 2 response. `soif` turns model +
tokens + hosting assumptions into an honest **low / mid / high** water estimate with a fully
documented, versioned methodology ([METHODOLOGY.md](METHODOLOGY.md)).

- Pure Python, **zero runtime dependencies**, MIT-licensed.
- Library API, CLI, SDK response adapters (OpenAI / Anthropic), a Claude Code hook, and
  water-aware model routing for agent graphs.

## Install

```bash
pip install soif            # from a checkout: pip install .
pip install "soif[tokenizers]"   # optional: exact token counts via tiktoken
```

## Quick start

```python
import soif

est = soif.estimate("gpt-4o", prompt="Explain retrieval-augmented generation.")
print(est.humanize())
# ~1.15 mL of water (23.0 drops); range 0.113 mL - 12.46 mL

est.total_ml.mid        # millilitres, mid scenario
est.onsite_ml           # cooling-tower evaporation at the data center
est.offsite_ml          # water consumed generating the electricity
est.embodied_ml         # amortised manufacturing (chips, servers, buildings)
est.assumptions         # every default the estimate leaned on, spelled out
```

The **accurate path** is to feed real token usage from an API response — this captures
actual output length, reasoning ("thinking") tokens, and cache hits:

```python
response = client.chat.completions.create(...)   # OpenAI — or Anthropic messages.create
est = soif.from_response(response)
```

Reasoning models drink more — thinking tokens are output tokens:

```python
soif.estimate("gpt-5", output_tokens=500, reasoning_effort="high")
soif.estimate("o3", output_tokens=500, reasoning_tokens=8000)   # from real usage
```

Control the hosting scenario:

```python
soif.estimate("llama-3.1-70b", output_tokens=500,
              provider="aws", region="nordics",      # presets
              include_embodied=False)                # operational water only
soif.estimate("my-fine-tune", active_params_b=8, wue=0.2, pue=1.12, ewif=0.4)
```

## CLI

```bash
soif estimate "why is the sky blue?" --model claude-sonnet-4-5
soif estimate -m gpt-4o -i 1200 -o 500 --json
soif compare gpt-4o gpt-4o-mini gemini-2.5-flash claude-haiku-4-5 -o 500
soif models
```

## Agent graphs: metering and minimising water across a chain

Two primitives make water a first-class optimization target in agentic pipelines
(LangGraph, hand-rolled DAGs, anything):

**1. `Meter` — accumulate across nodes, with a soft budget:**

```python
meter = soif.Meter(budget_ml=50)

def summarize_node(state):
    resp = client.chat.completions.create(model=state["model"], ...)
    meter.record(soif.from_response(resp))
    if meter.over_budget:
        state["model"] = "gpt-4o-mini"   # degrade later hops
    return state

print(meter.summary())
# 7 call(s): ~18.2 mL of water (3.7 teaspoons); range ... — within budget (18.2/50.0 mL)
```

**2. `soif.optimize` — route each node to the least-thirsty capable model:**

```python
from soif import optimize

optimize.pick_model(
    ["gpt-4o", "gpt-4o-mini", "gemini-2.5-flash", "claude-sonnet-4-5"],
    min_tier="small",              # capability floor for this node
    input_tokens=2000, output_tokens=300,
)
# -> 'gemini-2.5-flash'... whichever mid-scenario estimate is lowest

optimize.savings("claude-opus-4", "claude-haiku-4-5", output_tokens=500)
# {'baseline_ml': ..., 'alternative_ml': ..., 'saved_ml': ..., 'saved_pct': ...}
```

Model choice is the big lever (~30× between tiers); after that: shorter outputs, prompt
caching, modest reasoning effort, and low-water regions/providers.

## Claude Code hook

Get a water read-out for every session, computed from the transcript's *real* token usage
— see [integrations/claude-code](integrations/claude-code/README.md). In short, add to
`.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      { "hooks": [{ "type": "command", "command": "soif claude-hook" }] }
    ]
  }
}
```

After each turn: `soif: this session used ~4.31 mL of water (0.9 teaspoons); range ... across 12 model call(s).`

## How it works (short version)

```
E_it       = tokens × Wh-per-token(model tier)        # server energy
E_facility = E_it × PUE                               # + cooling/power overhead
W_onsite   = E_it × WUE                               # cooling evaporation
W_offsite  = E_facility × EWIF                        # power-plant water
W_total    = (W_onsite + W_offsite) × lifecycle       # + embodied (optional)
```

Every factor is a *(low, mid, high)* triple propagated end-to-end, so the range reflects
genuine uncertainty rather than false precision. Factors are versioned
(`soif.factors.FACTORS_VERSION`) and calibrated against Google's measured Gemini numbers,
Epoch AI's GPT-4o analysis, Mistral's Large 2 LCA, and Ren et al.'s methodology.
**Read [METHODOLOGY.md](METHODOLOGY.md) before quoting numbers** — these are estimates,
not measurements.

## Contributing

Factor updates (new disclosures, better WUE/PUE/EWIF data, new models) are the most
valuable contributions — please include sources. `pip install -e ".[dev]" && pytest && ruff check .`

## License

MIT
