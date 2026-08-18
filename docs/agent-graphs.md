# Water-aware agent graphs

In a chain of agent prompts, water is like latency or cost: a per-node quantity you can
**meter**, **budget**, and **minimise**. Model choice is the big lever — tiers differ by
roughly 30× per token — followed by shorter outputs, prompt caching, modest reasoning
effort, and low-water hosting.

Two primitives, framework-agnostic (LangGraph, hand-rolled DAGs, anything):

## 1. Meter every node

```python
import soif

meter = soif.Meter(budget_ml=50)

def node(state):
    resp = client.chat.completions.create(model=state["model"], ...)
    meter.record(soif.from_response(resp))     # real usage, not guesses
    if meter.over_budget:
        state["model"] = "gpt-4o-mini"         # degrade later hops
    return state
```

The budget is advisory — `soif` never raises — so your graph decides how to react:
switch tiers, truncate context, cache harder, or stop early. `meter.total` is a merged
`WaterEstimate`, so the full on-site/off-site/embodied breakdown survives aggregation.

## 2. Route each node to the least-thirsty capable model

```python
from soif import optimize

model = optimize.pick_model(
    ["gpt-4o", "gpt-4o-mini", "gemini-2.5-flash", "claude-sonnet-4-5"],
    min_tier="small",                  # capability floor for this node
    input_tokens=2000, output_tokens=300,
)
```

`min_tier` encodes "this node needs at least X-grade capability"; within that constraint
the ranking minimises mid-scenario water for the node's expected workload. Quantify a
routing decision with:

```python
optimize.savings("claude-opus-4", "claude-haiku-4-5", output_tokens=500)
# {'baseline_ml': 2.86, 'alternative_ml': 0.71, 'saved_ml': 2.14, 'saved_pct': 75.0}
```

## Full example

See [`examples/agent_graph.py`](https://github.com/Unchained-Labs/soif/blob/main/examples/agent_graph.py)
— a three-node route → research → synthesize pipeline that picks per-node models, meters
real usage shapes, and reports against a 25 mL budget. It runs offline (no API keys).
