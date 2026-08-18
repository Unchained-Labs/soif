from soif import registry
from soif.tokens import approx_tokens


def test_approx_tokens_heuristic():
    assert approx_tokens("") == 0
    assert approx_tokens("word") >= 1
    n = approx_tokens("hello world " * 100)
    assert 150 < n < 450


def test_longest_match_wins():
    assert registry.resolve("gpt-4o-mini-2024-07-18").tier == "small"
    assert registry.resolve("gpt-4o-2024-08-06").tier == "large"
    assert registry.resolve("openai/gpt-5-nano").tier == "nano"


def test_separator_normalisation():
    assert registry.resolve("Claude Opus 4.1").tier == "frontier"
    assert registry.resolve("deepseek_v3").tier == "medium"


def test_unknown_returns_none():
    assert registry.resolve("totally-made-up") is None
