"""Water-aware agent graph: meter every node, route models by water budget.

Runs offline (no API keys) by simulating usage payloads; swap the fake
calls for real SDK calls and `soif.from_response(...)` in production.
"""

import soif
from soif import optimize

# A three-node pipeline: route -> research -> synthesize.
# Each node declares a capability floor; the router picks the least-thirsty
# candidate that satisfies it.
CANDIDATES = ["claude-opus-4", "claude-sonnet-4-5", "claude-haiku-4-5", "gemini-2.5-flash"]

NODES = [
    {"name": "route", "min_tier": "small", "in": 400, "out": 50},
    {"name": "research", "min_tier": "medium", "in": 6000, "out": 1200},
    {"name": "synthesize", "min_tier": "large", "in": 3000, "out": 800},
]

meter = soif.Meter(budget_ml=25)

for node in NODES:
    model = optimize.pick_model(
        CANDIDATES,
        min_tier=node["min_tier"],
        input_tokens=node["in"],
        output_tokens=node["out"],
    )
    # ... call the model here; then record real usage:
    fake_usage = {"input_tokens": node["in"], "output_tokens": node["out"]}
    est = meter.record(soif.from_usage(fake_usage, model=model))
    print(f"{node['name']:<11} -> {model:<18} {est.humanize()}")

print()
print(meter.summary())

saved = optimize.savings("claude-opus-4", "claude-haiku-4-5", output_tokens=500)
print(
    f"\nRouting a 500-token step from opus to haiku saves "
    f"{saved['saved_ml']:.2f} mL/call ({saved['saved_pct']:.0f}%)."
)
