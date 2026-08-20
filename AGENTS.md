# AGENTS.md

Guidance for AI agents — both for **working on this repo** and for **using soif from an
agentic stack**.

## What soif is

Python library + CLI estimating the freshwater (mL) consumed to serve an LLM response.
All results are `(low, mid, high)` scenario ranges. PyPI: `soif-llm` (module `soif`).
Docs: https://unchained-labs.github.io/soif/ · Methodology: `METHODOLOGY.md` (read it
before quoting numbers — these are estimates, not measurements).

## Using soif from an agent

Rules of thumb:

- **Prefer real usage over guesses**: when a response object is available, use
  `soif.from_response(response)` / `soif.from_usage(usage, model=...)` — they handle
  reasoning tokens and cache reads without double counting. Use
  `soif.estimate(model, prompt=...)` only for what-if questions.
- **Always report the range**, never the mid value alone ("~1.4 mL, range 0.1–15 mL").
- **Quote assumptions**: `est.assumptions` lists every default the estimate leaned on.
- Estimates are **additive** (`a + b`); use `soif.Meter(budget_ml=...)` to accumulate
  across agent-graph nodes and react to a soft budget.
- To minimise water in a pipeline, route steps with
  `soif.optimize.pick_model(candidates, min_tier=...)` — model tier is a ~30× lever.
- Unknown model names fall back to the "large" tier with an explicit assumption; pass
  `tier=` or `active_params_b=` when you know better.

Cheat sheet:

```python
import soif
soif.estimate("gpt-4o", input_tokens=1200, output_tokens=500)      # what-if
soif.from_response(resp)                                           # accurate
soif.estimate("o3", output_tokens=500, reasoning_tokens=8000)      # thinking counts
soif.Meter(budget_ml=50)                                           # graph accounting
soif.optimize.pick_model([...], min_tier="small")                  # water routing
```

CLI: `soif estimate|compare|models|claude-hook` (`--json` everywhere).

Integrations:

- **MCP server**: https://github.com/Unchained-Labs/soif-mcp — exposes these
  capabilities as MCP tools for Claude Code/Desktop, Cursor, and agent frameworks.
- **Claude Code hook**: `integrations/claude-code/` — per-session water read-outs
  from real transcript usage. A ready-made skill lives in `.claude/skills/soif/`.

## Working on this repo

- Layout: `src/soif/` (library), `tests/`, `docs/` (MkDocs → GitHub Pages),
  `integrations/claude-code/`, `examples/`.
- Setup: `pip install -e ".[dev]"` · Test: `pytest -q` · Lint: `ruff check .`
  (line length 100). CI runs both on Python 3.10–3.13; keep it green.
- **Zero runtime dependencies** is a feature — don't add any to the core package.
- All physical constants live in `src/soif/factors.py` as `(low, mid, high)` triples
  with sources in `METHODOLOGY.md`. Changing any factor requires: bump
  `FACTORS_VERSION`, update the METHODOLOGY tables and sources, keep the calibration
  tests in `tests/test_estimator.py` passing.
- New model names go in `src/soif/registry.py` (longest-substring matching; tier by
  *active* params — MoE models count activated experts only).
- Never present outputs as measurements; keep uncertainty ranges end-to-end.
- Releases: tag `v*` → `release.yml` publishes `soif-llm` to PyPI via Trusted
  Publishing. Docs deploy from `docs.yml` on push to main.
