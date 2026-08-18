import pytest

import soif
from soif import factors
from soif.estimator import SoifError


def test_known_model_resolves_tier_and_provider():
    est = soif.estimate("gpt-4o", input_tokens=1000, output_tokens=500)
    assert est.tier == "large"
    assert est.provider == "azure"
    assert est.total_ml.low < est.total_ml.mid < est.total_ml.high


def test_prefixed_and_dated_model_names_match():
    est = soif.estimate("us.anthropic.claude-sonnet-4-5-20250929-v1:0", output_tokens=100)
    assert est.tier == "large"
    assert est.provider == "aws"


def test_unknown_model_falls_back_with_assumption():
    est = soif.estimate("mystery-llm-9000", output_tokens=100)
    assert est.tier == "large"
    assert any("unknown model" in a for a in est.assumptions)


def test_tier_from_active_params():
    est = soif.estimate("custom", active_params_b=8, output_tokens=100)
    assert est.tier == "small"


def test_gemini_flash_mid_estimate_near_google_published_number():
    # Google's measured median Gemini Apps prompt: 0.24 Wh, 0.26 mL
    # (operational, on-site scope). Our operational mid for a small-tier
    # model on GCP with a 500-token response should land within ~3x.
    est = soif.estimate("gemini-2.5-flash", input_tokens=500, output_tokens=500,
                        include_embodied=False)
    operational = est.total_ml.mid
    assert 0.08 < operational < 0.9
    assert 0.05 < est.energy_facility_wh.mid < 0.8


def test_uncertainty_range_brackets_literature_for_frontier():
    # Published per-response figures span ~0.3 mL (Google) to 45 mL
    # (Mistral lifecycle). A frontier estimate's low/high should span most
    # of that range.
    est = soif.estimate("claude-opus-4", input_tokens=1000, output_tokens=500)
    assert est.total_ml.low < 1.0
    assert est.total_ml.high > 10.0


def test_reasoning_tokens_increase_water():
    base = soif.estimate("o3", input_tokens=100, output_tokens=500)
    thinking = soif.estimate("o3", input_tokens=100, output_tokens=500,
                             reasoning_tokens=5000)
    assert thinking.total_ml.mid > base.total_ml.mid * 5


def test_reasoning_effort_presets():
    est = soif.estimate("gpt-5", output_tokens=500, reasoning_effort="high")
    assert est.reasoning_tokens == 5000


def test_input_tokens_cheaper_than_output_tokens():
    inp = soif.estimate("gpt-4o", input_tokens=1000, output_tokens=0)
    out = soif.estimate("gpt-4o", input_tokens=0, output_tokens=1000)
    assert out.total_ml.mid > inp.total_ml.mid * 5


def test_embodied_toggle():
    with_e = soif.estimate("gpt-4o", output_tokens=500)
    without = soif.estimate("gpt-4o", output_tokens=500, include_embodied=False)
    assert without.embodied_ml.mid == 0
    assert with_e.total_ml.mid > without.total_ml.mid


def test_prompt_text_estimates_tokens_and_default_output():
    est = soif.estimate("gpt-4o-mini", prompt="hello " * 200)
    assert est.input_tokens > 100
    assert est.output_tokens == factors.DEFAULT_OUTPUT_TOKENS


def test_custom_wue_pue_ewif_override():
    est = soif.estimate("gpt-4o", output_tokens=1000, wue=0.0, ewif=0.0,
                        include_embodied=False)
    assert est.total_ml.mid == 0.0


def test_estimates_are_additive():
    a = soif.estimate("gpt-4o", input_tokens=100, output_tokens=200)
    b = soif.estimate("gpt-4o-mini", input_tokens=300, output_tokens=400)
    total = a + b
    assert total.calls == 2
    assert total.input_tokens == 400
    assert total.total_ml.mid == pytest.approx(a.total_ml.mid + b.total_ml.mid)


def test_invalid_inputs_raise():
    with pytest.raises(SoifError):
        soif.estimate("gpt-4o", tier="galactic")
    with pytest.raises(SoifError):
        soif.estimate("gpt-4o", provider="cornerstore")
    with pytest.raises(SoifError):
        soif.estimate("gpt-4o", region="atlantis")
    with pytest.raises(SoifError):
        soif.estimate("gpt-4o", reasoning_effort="ultra")


def test_to_dict_shape():
    d = soif.estimate("gpt-4o", output_tokens=100).to_dict()
    assert set(d["water_ml"]) == {"total", "onsite_cooling", "offsite_electricity", "embodied"}
    assert d["factors_version"] == factors.FACTORS_VERSION
