import soif


def test_openai_chat_completions_usage_dict():
    usage = {
        "prompt_tokens": 1200,
        "completion_tokens": 600,
        "completion_tokens_details": {"reasoning_tokens": 100},
        "prompt_tokens_details": {"cached_tokens": 200},
    }
    est = soif.from_usage(usage, model="gpt-4o")
    assert est.input_tokens == 1000  # cached part split out
    assert est.cached_tokens == 200
    assert est.output_tokens == 500
    assert est.reasoning_tokens == 100
    assert est.total_ml.mid > 0


def test_anthropic_usage_dict():
    usage = {
        "input_tokens": 800,
        "output_tokens": 400,
        "cache_read_input_tokens": 5000,
        "cache_creation_input_tokens": 100,
    }
    est = soif.from_usage(usage, model="claude-sonnet-4-5")
    assert est.input_tokens == 900
    assert est.cached_tokens == 5000
    assert est.tier == "large"


def test_from_response_duck_typed():
    class Usage:
        input_tokens = 100
        output_tokens = 50

    class Response:
        model = "claude-haiku-4-5"
        usage = Usage()

    est = soif.from_response(Response())
    assert est.model == "claude-haiku-4-5"
    assert est.tier == "medium"


def test_energy_not_double_counted_for_reasoning():
    # reasoning tokens are inside completion_tokens; splitting them out must
    # not change the energy estimate.
    plain = soif.from_usage({"prompt_tokens": 0, "completion_tokens": 1000}, model="o3")
    split = soif.from_usage(
        {"prompt_tokens": 0, "completion_tokens": 1000,
         "completion_tokens_details": {"reasoning_tokens": 900}},
        model="o3",
    )
    assert split.total_ml.mid == plain.total_ml.mid
    assert split.output_tokens == 100
    assert split.reasoning_tokens == 900
