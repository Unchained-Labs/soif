"""soif — estimate the water footprint of LLM prompts.

*soif* is French for "thirst". The library answers "how much water did this
answer cost?" the way cost calculators answer it in dollars: per model, per
token, with honest uncertainty ranges and a documented methodology.

Quick start::

    import soif

    est = soif.estimate("gpt-4o", prompt="Explain RAG in one paragraph.")
    print(est.humanize())          # ~1.4 mL of water (0.3 teaspoons); range ...

    # Accurate path: feed real usage from an API response
    est = soif.from_response(openai_response)

    # Agent graphs: accumulate across nodes, route by water
    meter = soif.Meter(budget_ml=50)
    meter.record(est)
    best = soif.optimize.pick_model(["gpt-4o", "gpt-4o-mini"], min_tier="small")
"""

from soif import optimize
from soif._triple import Triple
from soif.adapters import from_response, from_usage
from soif.estimator import SoifError, WaterEstimate, estimate
from soif.meter import Meter
from soif.tokens import approx_tokens

__version__ = "0.1.0"

__all__ = [
    "Meter",
    "SoifError",
    "Triple",
    "WaterEstimate",
    "__version__",
    "approx_tokens",
    "estimate",
    "from_response",
    "from_usage",
    "optimize",
]
