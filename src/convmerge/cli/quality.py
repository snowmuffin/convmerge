"""filter and decontam commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from convmerge.cli._common import positive_int as _positive_int


def _fraction(value: str) -> float:
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not 0 < f <= 1:
        raise argparse.ArgumentTypeError(f"expected a fraction in (0, 1], got {value}")
    return f


def _non_negative(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {n}")
    return n


def _script_share(value: str) -> tuple[str, float]:
    script, sep, share = value.partition("=")
    if not sep or not script:
        raise argparse.ArgumentTypeError(f"expected SCRIPT=SHARE (e.g. hangul=0.3), got {value!r}")
    return script.strip(), _fraction(share)


def _add_filter(sub: argparse._SubParsersAction) -> None:
    from convmerge.quality import DEFAULT_RULES, RULES

    p = sub.add_parser(
        "filter",
        help="Drop refusals, empty answers, loops, and bad preference pairs (rule based)",
        description="Check every row against deterministic quality rules and print a JSON "
        "report; -o keeps the rows that pass, byte-for-byte. Rules on by default: "
        f"{', '.join(DEFAULT_RULES)}. Others: "
        f"{', '.join(r for r in RULES if r not in DEFAULT_RULES)}.",
    )
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, default=None, help="Write rows that pass here")
    p.add_argument("--rejects", type=Path, default=None, help="Write the other rows here")
    p.add_argument(
        "--enable", action="append", default=[], metavar="RULE", choices=list(RULES),
        help="Turn a rule on (repeatable)",
    )  # fmt: skip
    p.add_argument(
        "--disable", action="append", default=[], metavar="RULE", choices=list(RULES),
        help="Turn a rule off (repeatable)",
    )  # fmt: skip
    p.add_argument(
        "--min-answer-chars", type=_non_negative, default=1,
        help="empty_answer: shortest final answer kept (default 1)",
    )  # fmt: skip
    p.add_argument("--min-chars", type=_non_negative, default=None,
                   help="length: shortest answer text kept (turns the rule on)")  # fmt: skip
    p.add_argument("--max-chars", type=_non_negative, default=None,
                   help="length: longest answer text kept (turns the rule on)")  # fmt: skip
    p.add_argument(
        "--repetition-max", type=_fraction, default=0.7,
        help="repetition: share of repeated word 10-grams that marks a loop (default 0.7)",
    )  # fmt: skip
    p.add_argument("--slop-max", type=_positive_int, default=3,
                   help="slop: stock phrases per row that drop it (default 3)")  # fmt: skip
    p.add_argument(
        "--min-script", action="append", type=_script_share, default=[], metavar="SCRIPT=SHARE",
        help="script: least share of letters in SCRIPT (hangul, latin, han, kana, cyrillic); "
        "turns the rule on",
    )  # fmt: skip
    p.add_argument(
        "--rules-file", type=Path, default=None, metavar="YAML",
        help="Extra refusal / slop phrases and regex rules (see docs/quality.md)",
    )  # fmt: skip
    p.add_argument(
        "--workers", type=_positive_int, default=1, metavar="N",
        help="Check rows with N processes (default 1). Output and report are identical "
        "to a single-process run",
    )  # fmt: skip


def _cmd_filter(args: argparse.Namespace) -> None:
    from convmerge.convert import REPORT_VERSION
    from convmerge.quality import FilterSpec, FilterStats, filter_jsonl

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    try:
        spec = FilterSpec.from_options(
            enable=args.enable, disable=args.disable, min_chars=args.min_chars,
            max_chars=args.max_chars, min_script=dict(args.min_script),
            rules_file=args.rules_file, min_answer_chars=args.min_answer_chars,
            repetition_max=args.repetition_max, slop_max=args.slop_max,
        )  # fmt: skip
    except (ImportError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    stats = FilterStats()
    filter_jsonl(
        args.input, spec=spec, output=args.output, rejects=args.rejects, stats=stats,
        workers=args.workers,
    )  # fmt: skip
    report = {"version": REPORT_VERSION, **stats.to_report()}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    hit = {k: v for k, v in stats.rules.items() if v}
    if hit:
        summary = ", ".join(f"{k}={v:,}" for k, v in hit.items())
        print(f"matched: {summary}", file=sys.stderr)
    if stats.unreadable:
        print(f"warning: {stats.unreadable:,} rows are not in a format convmerge reads",
              file=sys.stderr)  # fmt: skip
    if stats.invalid_json:
        print(f"warning: dropped {stats.invalid_json:,} invalid JSON lines "
              f"(first at line {stats.first_invalid_line})", file=sys.stderr)  # fmt: skip
    for warning in stats.warnings():
        print(f"warning: {warning}", file=sys.stderr)
    verb = f"kept {stats.kept:,} -> {args.output}" if args.output else f"would keep {stats.kept:,}"
    print(f"{verb}; rejected {stats.rejected:,} of {stats.rows:,}", file=sys.stderr)


def _add_decontam(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "decontam",
        help="Drop rows that share a word n-gram with an evaluation set",
        description="Index the evaluation sets' word n-grams and drop training rows that "
        "contain one (prompts only unless --check all). Prints a JSON report; -o keeps "
        "the clean rows, byte-for-byte. hf: sources need convmerge[fetch-hf].",
    )
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument(
        "--against", action="append", required=True, metavar="EVAL",
        help="Evaluation set: a JSONL file or hf:REPO[:CONFIG[:SPLIT]] (split defaults to "
        "test); repeatable",
    )  # fmt: skip
    p.add_argument("--output", "-o", type=Path, default=None, help="Write clean rows here")
    p.add_argument("--rejects", type=Path, default=None, help="Write contaminated rows here")
    p.add_argument("--ngram", type=_positive_int, default=13, help="Words per n-gram (13)")
    p.add_argument(
        "--min-tokens", type=_positive_int, default=8,
        help="Shorter evaluation passages must appear whole; below this they are skipped (8)",
    )  # fmt: skip
    p.add_argument(
        "--check", choices=("prompts", "all"), default="prompts",
        help="Training text to check: system/user turns (default) or all turns",
    )  # fmt: skip
    p.add_argument(
        "--fields", default=None, metavar="A,B",
        help="Evaluation fields to read (default: every string field of a row)",
    )  # fmt: skip
    p.add_argument("--hf-token", default=None, help="Token for gated evaluation sets (or HF_TOKEN)")


def _cmd_decontam(args: argparse.Namespace) -> None:
    import os

    from convmerge.convert import REPORT_VERSION
    from convmerge.decontam import DecontamStats, EvalSource, build_index, decontaminate_jsonl

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip()) if args.fields else None
    sources = [EvalSource(spec, fields) for spec in args.against]
    token = args.hf_token or os.environ.get("HF_TOKEN") or None
    try:
        for source in sources:
            source.hub  # noqa: B018 - validates hf: specs before any download
        index = build_index(sources, ngram=args.ngram, min_tokens=args.min_tokens, token=token)
    except (ImportError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except (OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    stats = DecontamStats()
    decontaminate_jsonl(args.input, index, check=args.check, output=args.output,
                        rejects=args.rejects, stats=stats)  # fmt: skip
    report = {"version": REPORT_VERSION, **stats.to_report()}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for name, counts in stats.eval_sets.items():
        print(f"{name}: {counts['passages']:,} passages, matched {counts['matched']:,} rows",
              file=sys.stderr)  # fmt: skip
    verb = f"kept {stats.kept:,} -> {args.output}" if args.output else f"would keep {stats.kept:,}"
    print(f"{verb}; contaminated {stats.contaminated:,} of {stats.rows:,}", file=sys.stderr)
