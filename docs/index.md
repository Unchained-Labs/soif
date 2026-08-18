# soif 💧

> *soif* — French for **thirst**. Estimate the water footprint of LLM prompts, the way you
> estimate their cost.

Every LLM answer evaporates real freshwater: data-center cooling towers (on-site) and the
power plants feeding them (off-site) both consume it. Published per-prompt figures span
two orders of magnitude — Google measured **0.26 mL** per median Gemini prompt, while
Mistral's lifecycle analysis reports **45 mL** per 400-token Large 2 response. `soif`
turns model + tokens + hosting assumptions into an honest **low / mid / high** water
estimate with a documented, versioned [methodology](methodology.md).

## Install

```bash
pip install soif-llm                  # imports as `soif`
pip install "soif-llm[tokenizers]"    # optional: exact token counts via tiktoken
```

!!! note
    The PyPI distribution is **`soif-llm`** (the bare name was taken); the Python module
    is `soif`.

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

The **accurate path** is to feed real token usage from an API response — actual output
length, reasoning ("thinking") tokens, and cache hits included:

```python
response = client.chat.completions.create(...)   # OpenAI or Anthropic
est = soif.from_response(response)
```

Reasoning models drink more — thinking tokens are output tokens:

```python
soif.estimate("gpt-5", output_tokens=500, reasoning_effort="high")
soif.estimate("o3", output_tokens=500, reasoning_tokens=8000)   # from real usage
```

## CLI

```bash
soif estimate "why is the sky blue?" --model claude-sonnet-4-5
soif estimate -m gpt-4o -i 1200 -o 500 --json
soif compare gpt-4o gpt-4o-mini gemini-2.5-flash claude-haiku-4-5 -o 500
soif models
```

## Where next

- [Methodology](methodology.md) — the model, factors, sources, and limits. Read it
  before quoting numbers.
- [API reference](api.md) — the full Python surface.
- [Agent graphs](agent-graphs.md) — meter a pipeline and route models by water.
- [Claude Code hook](claude-code.md) — per-session water read-outs from real usage.
