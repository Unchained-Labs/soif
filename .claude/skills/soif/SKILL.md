---
name: soif
description: Estimate the water footprint of LLM prompts, sessions, or pipelines in millilitres, compare models by water use, or route agent steps to the least-thirsty capable model. Use when the user asks how much water an LLM call/answer/session used or would use, asks about the environmental or water cost of AI usage, wants to compare models on water/sustainability, or wants to reduce the water footprint of an agent pipeline.
---

# soif — LLM water-footprint estimation

Use the `soif` Python library (`pip install soif-llm`, zero dependencies) or its CLI.
If neither is installed, install with `pip install soif-llm`.

## How to answer water questions

1. **Have a real usage/response object?** Use the accurate path:
   ```python
   import soif
   est = soif.from_usage({"input_tokens": 800, "output_tokens": 400}, model="claude-sonnet-4-5")
   ```
   Accepts OpenAI (prompt_tokens/completion_tokens + details) and Anthropic
   (input_tokens/output_tokens, cache_read_input_tokens) shapes; reasoning and cached
   tokens are handled without double counting.

2. **What-if question?** `soif.estimate(model, prompt=... | input_tokens=..., output_tokens=...)`.
   Reasoning models: pass `reasoning_tokens=` (real) or `reasoning_effort="low|medium|high"`.
   Hosting overrides: `provider="aws|azure|gcp|average"`,
   `region="world|us|eu|france|nordics|asia|renewable"`, `include_embodied=False` for
   operational water only.

3. **Whole Claude Code session?** `soif claude-hook --transcript <session.jsonl> < /dev/null`
   sums real per-message usage from the transcript.

4. **Compare or route models?**
   ```python
   from soif import optimize
   optimize.rank(["gpt-4o", "gpt-4o-mini"], output_tokens=500)
   optimize.pick_model(candidates, min_tier="small")     # least-thirsty capable model
   ```
   Accumulate across pipeline nodes with `soif.Meter(budget_ml=...)`.

CLI equivalents: `soif estimate -m MODEL -i N -o N [--json]`, `soif compare M1 M2 ...`,
`soif models`.

## Reporting rules (important)

- Results are `(low, mid, high)` **scenario ranges**. Always report the mid value WITH
  its range (`est.humanize()` does this) — never the mid alone as a precise fact.
- These are **estimates from public data, not measurements**. Mention `est.assumptions`
  when they materially affect the answer (unknown model, assumed output length).
- Water splits into on-site cooling (`onsite_ml`), electricity generation
  (`offsite_ml`), and embodied manufacturing (`embodied_ml`) — cite the breakdown when
  asked "where does the water go?".
- Helpful anchors: Google measured ~0.26 mL per median Gemini prompt (operational);
  Mistral's lifecycle analysis reports ~45 mL per 400-token Large 2 response. The
  literature genuinely spans that range; soif's ranges bracket it.
- Methodology and sources: https://unchained-labs.github.io/soif/methodology/
