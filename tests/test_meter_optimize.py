import pytest

import soif
from soif.optimize import pick_model, rank, savings


def test_meter_accumulates_and_budgets():
    meter = soif.Meter(budget_ml=1.0)
    assert meter.total is None
    assert not meter.over_budget

    meter.record(soif.estimate("claude-opus-4", input_tokens=2000, output_tokens=2000))
    assert meter.total_ml > 1.0
    assert meter.over_budget
    assert meter.remaining_ml == 0.0
    assert "OVER" in meter.summary()


def test_rank_orders_by_water():
    ranked = rank(["gpt-4o", "gpt-4o-mini", "gpt-5"], output_tokens=500)
    mls = [r.ml for r in ranked]
    assert mls == sorted(mls)
    assert ranked[0].model == "gpt-4o-mini"


def test_min_tier_filters_candidates():
    ranked = rank(["gpt-4o", "gpt-4o-mini"], min_tier="large")
    assert [r.model for r in ranked] == ["gpt-4o"]


def test_pick_model_and_savings():
    assert pick_model(["claude-opus-4", "claude-haiku-4-5"]) == "claude-haiku-4-5"
    s = savings("claude-opus-4", "claude-haiku-4-5", output_tokens=500)
    assert s["saved_ml"] > 0
    assert 0 < s["saved_pct"] < 100


def test_pick_model_no_candidates_raises():
    with pytest.raises(ValueError):
        pick_model(["gpt-4o-mini"], min_tier="frontier")
