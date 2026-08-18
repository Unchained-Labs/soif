# Methodology

`soif` estimates the freshwater **consumed** (evaporated or otherwise removed from the
local watershed — not merely withdrawn and returned) to serve an LLM response. This
document explains the model, the factors, their sources, and the limits of the whole
exercise. Factor values live in [`src/soif/factors.py`](src/soif/factors.py) and are
versioned via `FACTORS_VERSION`.

## 1. The model

Following the operational-water methodology of Ren et al. ([*Making AI Less "Thirsty"*,
arXiv:2304.03271](https://arxiv.org/abs/2304.03271)), with an optional embodied adder:

```
E_it       = (output_tokens + reasoning_tokens
              + 0.1 × input_tokens + 0.01 × cached_tokens) / 1000 × Wh_per_1k(tier)
E_facility = E_it × PUE
W_onsite   = E_it × WUE            # cooling-tower / evaporative cooling at the DC
W_offsite  = E_facility × EWIF     # water consumed generating the electricity
W_embodied = (W_onsite + W_offsite) × (lifecycle_multiplier − 1)
W_total    = W_onsite + W_offsite + W_embodied
```

Units work out neatly: **1 L/kWh ≡ 1 mL/Wh**, so Wh × (L/kWh) yields millilitres.

Every factor is a **(low, mid, high)** scenario triple and the triples are multiplied
through, so the reported range is a best-case/central/worst-case *scenario spread*, not a
statistical confidence interval. This is deliberate: public per-prompt figures disagree by
~100× and pretending otherwise would be false precision.

## 2. Energy per token

Direct measurement is impossible from outside a provider, so `soif` buckets models into
five **tiers by active parameter count** (MoE models count activated experts only), each
with a server-side (IT) energy range per 1000 **output** tokens:

| Tier | Active params | Wh / 1k output tokens (low/mid/high) | Examples |
|---|---|---|---|
| nano | < 3B | 0.02 / 0.06 / 0.15 | gemini-flash-lite, gpt-5-nano |
| small | 3–15B | 0.06 / 0.20 / 0.50 | gpt-4o-mini, gemini-flash, llama-8b |
| medium | 15–70B | 0.15 / 0.50 / 1.20 | claude-haiku, deepseek-v3 (~37B active) |
| large | 70–250B | 0.30 / 0.80 / 2.00 | gpt-4o, claude-sonnet, mistral-large |
| frontier | > 250B / premium | 0.80 / 2.00 / 5.00 | gpt-5, o3, claude-opus, grok |

Calibration anchors:

- **Epoch AI** estimates ~**0.3 Wh** for a typical GPT-4o query with ~500 output tokens
  (server power, ex-PUE) → 0.6 Wh/1k, inside our *large* band (0.3–2.0, mid 0.8).
  ([epoch.ai](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use))
- **Google** measured **0.24 Wh** per *median* Gemini Apps prompt in May 2025 — full stack,
  including idle machines and PUE ([arXiv:2508.15734](https://arxiv.org/abs/2508.15734)).
  A small-tier 500-token response at mid scenario gives 0.10 Wh IT / 0.11 Wh facility;
  the median production prompt plausibly sits between our small mid and high scenarios.
- Prefill (input) tokens are batched and compute-bound rather than memory-bound; we charge
  them **10%** of an output token, and **1%** for cache-read tokens. Reasoning/thinking
  tokens are decode work and are charged at the full output rate.

## 3. On-site water (WUE) and overhead (PUE)

WUE = litres evaporated on-site per kWh of IT energy; PUE = facility/IT energy. Provider
presets from public disclosures:

| Provider preset | WUE L/kWh (low/mid/high) | PUE (low/mid/high) | Basis |
|---|---|---|---|
| aws | 0.10 / 0.18 / 0.40 | 1.10 / 1.15 / 1.25 | AWS reported fleet WUE 0.15–0.18 |
| azure | 0.30 / 0.49 / 0.80 | 1.12 / 1.18 / 1.30 | Microsoft reported ~0.49 |
| gcp | 0.90 / 1.10 / 1.40 | 1.09 / 1.10 / 1.15 | Google fleet ~1.1 L/kWh, PUE 1.10 |
| average | 0.30 / 1.00 / 1.90 | 1.20 / 1.40 / 1.60 | industry surveys (Uptime ~1.56 PUE) |

OpenAI models default to `azure`, Anthropic to `aws`, Google to `gcp`, open-weight and
unknown models to `average`. Override with `provider=` or raw `wue=` / `pue=`.

## 4. Off-site water (EWIF)

Electricity generation consumes water (thermoelectric cooling, hydro reservoir
evaporation). Regional consumption-weighted factors (L/kWh) follow Ren et al. / Macknick
et al.; ranges are wide because grid mix and season dominate:

| Region | EWIF L/kWh (low/mid/high) |
|---|---|
| world (default) | 0.40 / 1.50 / 3.20 |
| us | 0.50 / 1.60 / 3.10 |
| eu | 0.30 / 1.20 / 2.50 |
| france | 0.50 / 1.50 / 3.00 |
| nordics | 0.05 / 0.30 / 0.90 |
| asia | 0.50 / 1.90 / 3.50 |
| renewable (matched wind/solar PPA) | 0.001 / 0.10 / 0.50 |

Note that "market-based" corporate accounting (buying renewable certificates) does not
remove the physical water use of the local grid; `soif` models physical (location-based)
water. Use `region="renewable"` only for genuinely co-located/matched supply.

## 5. Embodied water

Chip fabrication (ultra-pure water in fabs), server manufacturing, and data-center
construction consume water that should be amortised over the hardware's useful life.
Public data is thin; Mistral's Large 2 LCA — **45 mL per 400-token response**, lifecycle —
implies embodied + upstream shares far exceeding operational water, while Google's
operational figure is ~170× lower. `soif` applies a deliberately wide lifecycle
multiplier of **1.1× / 1.5× / 3.0×** on operational water, reported separately and
removable via `include_embodied=False`.

## 6. Cross-checks against the literature

For a ~500-token response, mid scenarios (embodied included):

| Case | soif mid (range) | Published |
|---|---|---|
| gemini-2.5-flash, gcp | ≈0.4 mL (0.08–2.5) | Google: 0.26 mL (operational, median prompt) |
| gpt-4o, azure/us | ≈1.4 mL (0.14–15) | Epoch: 0.3 Wh ⇒ ~0.5–1.5 mL depending on WUE/EWIF |
| mistral-large, france | ≈1.5 mL (0.2–16) | Mistral LCA: 45 mL (full lifecycle, incl. training amortisation) |
| GPT-3-era frontier, worst case | high tail 15–50 mL | Ren et al.: 10–50 mL per medium response |

The spread between Google and Mistral is largely **scope** (operational vs. full
lifecycle including amortised training, which `soif` does not include) and **methodology**
(measured medians vs. LCA attribution). `soif`'s ranges are designed to bracket the
defensible literature, with the mid scenario tracking measured operational figures plus a
moderate embodied adder.

## 7. What this is not

- **Not a measurement.** Only providers can measure; everything here is estimation from
  public data.
- **Training and R&D amortisation are excluded** (Mistral includes them; Google excludes
  them). They can dominate lifecycle numbers for low-traffic models.
- **Water stress is not weighted.** A millilitre in a water-stressed basin matters more
  than one in a rainy region; see "Not All Water Consumption Is Equal"
  ([arXiv:2506.22773](https://arxiv.org/abs/2506.22773)) for a stress-weighted approach —
  a good future extension.
- **Factors age fast.** Hardware efficiency improved ~33× year-over-year in Google's
  disclosure window. Check `FACTORS_VERSION`, and file an issue/PR when better data lands.

## Sources

- Li, Yang, Islam, Ren — *Making AI Less "Thirsty"* — [arXiv:2304.03271](https://arxiv.org/abs/2304.03271) / [CACM](https://cacm.acm.org/sustainability-and-computing/making-ai-less-thirsty/)
- Google — *Measuring the environmental impact of delivering AI at Google scale* — [arXiv:2508.15734](https://arxiv.org/abs/2508.15734) / [blog](https://cloud.google.com/blog/products/infrastructure/measuring-the-environmental-impact-of-ai-inference)
- Epoch AI — *How much energy does ChatGPT use?* — [epoch.ai](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use)
- Mistral AI — *Our contribution to a global environmental standard for AI* (Large 2 LCA, with ADEME/Carbone 4) — see [coverage](https://www.deeplearning.ai/the-batch/french-ai-startup-discloses-full-lifecycle-consumption-and-emissions-for-mistral-large-2)
- Macknick et al. — water consumption factors for electricity generation (NREL)
- Provider sustainability reports: AWS water positive updates (WUE 0.15–0.18), Microsoft (~0.49), Google (fleet WUE ~1.1, PUE 1.10)
