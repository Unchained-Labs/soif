"""soif command-line interface.

Subcommands:
  estimate     Estimate water for a model + prompt/token counts.
  compare      Rank several models on the same workload.
  models       List known models.
  claude-hook  Claude Code Stop-hook: report the session's water footprint.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from soif import __version__, factors, registry
from soif.estimator import SoifError, WaterEstimate, estimate
from soif.optimize import rank


def _print_estimate(est: WaterEstimate, as_json: bool) -> None:
    if as_json:
        print(json.dumps(est.to_dict(), indent=2))
        return
    t = est.total_ml
    print(f"model      {est.model or '(unspecified)'}  [tier={est.tier}, "
          f"provider={est.provider}, region={est.region}]")
    print(f"tokens     in={est.input_tokens} out={est.output_tokens}"
          + (f" reasoning={est.reasoning_tokens}" if est.reasoning_tokens else "")
          + (f" cached={est.cached_tokens}" if est.cached_tokens else ""))
    print(f"energy     {est.energy_it_wh.mid:.3f} Wh IT "
          f"({est.energy_facility_wh.mid:.3f} Wh at the meter)")
    print(f"water      {est.humanize()}")
    print(f"  on-site cooling        {est.onsite_ml.mid:8.3f} mL")
    print(f"  off-site electricity   {est.offsite_ml.mid:8.3f} mL")
    print(f"  embodied (lifecycle)   {est.embodied_ml.mid:8.3f} mL")
    print(f"  total (low/mid/high)   {t.low:.3f} / {t.mid:.3f} / {t.high:.3f} mL")
    for a in est.assumptions:
        print(f"note: {a}")


def _cmd_estimate(args: argparse.Namespace) -> int:
    est = estimate(
        args.model,
        prompt=args.prompt,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        reasoning_tokens=args.reasoning_tokens,
        cached_tokens=args.cached_tokens,
        reasoning_effort=args.reasoning_effort,
        tier=args.tier,
        active_params_b=args.params,
        provider=args.provider,
        region=args.region,
        include_embodied=not args.no_embodied,
    )
    _print_estimate(est, args.json)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    ranked = rank(
        args.models,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens or factors.DEFAULT_OUTPUT_TOKENS,
        min_tier=args.min_tier,
    )
    if args.json:
        print(json.dumps(
            [{"model": r.model, "tier": r.tier, "mid_ml": r.ml,
              "range_ml": [r.estimate.total_ml.low, r.estimate.total_ml.high]}
             for r in ranked],
            indent=2,
        ))
        return 0
    width = max((len(r.model) for r in ranked), default=10)
    for r in ranked:
        t = r.estimate.total_ml
        print(f"{r.model:<{width}}  {r.tier:<8}  {t.mid:8.3f} mL  "
              f"({t.low:.3f} - {t.high:.3f})")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    specs = registry.known_models()
    if args.json:
        print(json.dumps(
            [{"match": s.match, "tier": s.tier, "provider": s.provider, "region": s.region}
             for s in specs],
            indent=2,
        ))
        return 0
    for s in specs:
        note = f"  ({s.notes})" if s.notes else ""
        print(f"{s.match:<24} {s.tier:<9} {s.provider:<8} {s.region}{note}")
    return 0


# ---------------------------------------------------------------------------
# Claude Code Stop hook
# ---------------------------------------------------------------------------


def _iter_transcript_usage(transcript_path: str) -> dict[str, dict[str, Any]]:
    """Collect per-assistant-message usage from a Claude Code transcript.

    Transcripts are JSONL; a streamed assistant message can appear on
    several lines sharing one message id, so keep the last usage per id.
    """
    by_id: dict[str, dict[str, Any]] = {}
    with open(transcript_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "assistant":
                continue
            message = entry.get("message") or {}
            usage = message.get("usage")
            if not usage:
                continue
            msg_id = message.get("id") or entry.get("uuid") or str(len(by_id))
            by_id[msg_id] = {"model": message.get("model"), "usage": usage}
    return by_id


def _cmd_claude_hook(args: argparse.Namespace) -> int:
    from soif.adapters import from_usage

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}
    transcript_path = args.transcript or payload.get("transcript_path")
    if not transcript_path:
        print("soif: no transcript_path in hook payload", file=sys.stderr)
        return 0  # never block the agent over telemetry

    try:
        messages = _iter_transcript_usage(transcript_path)
    except OSError as exc:
        print(f"soif: cannot read transcript: {exc}", file=sys.stderr)
        return 0
    if not messages:
        return 0

    total: WaterEstimate | None = None
    for record in messages.values():
        est = from_usage(record["usage"], model=record["model"], region=args.region)
        total = est if total is None else total + est
    assert total is not None
    message = (
        f"soif: this session used {total.humanize()} "
        f"across {total.calls} model call(s)."
    )
    print(json.dumps({"systemMessage": message}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soif",
        description="Estimate the water footprint of LLM prompts.",
    )
    parser.add_argument("--version", action="version", version=f"soif {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--json", action="store_true", help="emit JSON")

    p_est = sub.add_parser("estimate", help="estimate one call")
    p_est.add_argument("prompt", nargs="?", help="prompt text (tokens approximated)")
    p_est.add_argument("--model", "-m", help="model name, e.g. gpt-4o, claude-sonnet-4-5")
    p_est.add_argument("--input-tokens", "-i", type=int, default=0)
    p_est.add_argument("--output-tokens", "-o", type=int, default=None)
    p_est.add_argument("--reasoning-tokens", type=int, default=0)
    p_est.add_argument("--cached-tokens", type=int, default=0)
    p_est.add_argument("--reasoning-effort",
                       choices=list(factors.REASONING_EFFORT_TOKENS_PER_OUTPUT))
    p_est.add_argument("--tier", choices=factors.TIER_ORDER)
    p_est.add_argument("--params", type=float, help="active parameters in billions")
    p_est.add_argument("--provider", choices=list(factors.PROVIDERS))
    p_est.add_argument("--region", choices=list(factors.REGIONS))
    p_est.add_argument("--no-embodied", action="store_true",
                       help="operational water only (exclude manufacturing)")
    add_common(p_est)
    p_est.set_defaults(func=_cmd_estimate)

    p_cmp = sub.add_parser("compare", help="rank models by water use for a workload")
    p_cmp.add_argument("models", nargs="+")
    p_cmp.add_argument("--input-tokens", "-i", type=int, default=1000)
    p_cmp.add_argument("--output-tokens", "-o", type=int, default=None)
    p_cmp.add_argument("--min-tier", choices=factors.TIER_ORDER)
    add_common(p_cmp)
    p_cmp.set_defaults(func=_cmd_compare)

    p_models = sub.add_parser("models", help="list known models")
    add_common(p_models)
    p_models.set_defaults(func=_cmd_models)

    p_hook = sub.add_parser(
        "claude-hook",
        help="Claude Code Stop-hook: reads hook JSON on stdin, reports session water use",
    )
    p_hook.add_argument("--transcript", help="transcript path (default: from hook payload)")
    p_hook.add_argument("--region", choices=list(factors.REGIONS), default=None)
    p_hook.set_defaults(func=_cmd_claude_hook)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SoifError as exc:
        print(f"soif: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
