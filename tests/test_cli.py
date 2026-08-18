import json

from soif.cli import main


def test_estimate_text(capsys):
    assert main(["estimate", "--model", "gpt-4o", "-i", "1000", "-o", "500"]) == 0
    out = capsys.readouterr().out
    assert "water" in out
    assert "on-site cooling" in out


def test_estimate_json(capsys):
    assert main(["estimate", "--model", "gemini-2.5-flash", "-o", "500", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["tier"] == "small"
    assert data["water_ml"]["total"]["mid"] > 0


def test_estimate_prompt_positional(capsys):
    assert main(["estimate", "hello world, please explain transformers",
                 "--model", "claude-sonnet-4-5"]) == 0
    assert "note:" in capsys.readouterr().out


def test_compare(capsys):
    assert main(["compare", "gpt-4o", "gpt-4o-mini", "-o", "500"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("gpt-4o-mini")


def test_models(capsys):
    assert main(["models"]) == 0
    assert "gpt-4o" in capsys.readouterr().out


def test_unknown_tier_is_clean_error(capsys):
    assert main(["estimate", "--model", "gpt-4o", "--reasoning-effort", "high",
                 "--region", "renewable"]) == 0


def test_claude_hook(tmp_path, capsys, monkeypatch):
    transcript = tmp_path / "t.jsonl"
    lines = [
        {"type": "user", "message": {"content": "hi"}},
        {"type": "assistant",
         "message": {"id": "msg_1", "model": "claude-sonnet-4-5",
                     "usage": {"input_tokens": 10, "output_tokens": 5}}},
        # streamed continuation of the same message: must not double count
        {"type": "assistant",
         "message": {"id": "msg_1", "model": "claude-sonnet-4-5",
                     "usage": {"input_tokens": 1000, "output_tokens": 500}}},
        {"type": "assistant",
         "message": {"id": "msg_2", "model": "claude-sonnet-4-5",
                     "usage": {"input_tokens": 2000, "output_tokens": 300,
                               "cache_read_input_tokens": 8000}}},
    ]
    transcript.write_text("\n".join(json.dumps(x) for x in lines))

    import io
    import sys
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        {"hook_event_name": "Stop", "transcript_path": str(transcript)})))
    assert main(["claude-hook"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "2 model call(s)" in out["systemMessage"]
    assert "water" in out["systemMessage"]


def test_claude_hook_missing_transcript_is_silent(monkeypatch, capsys):
    import io
    import sys
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    assert main(["claude-hook"]) == 0
