"""label command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from convmerge._text import count
from convmerge.cli._common import positive_int as _positive_int

DEFAULT_CACHE = Path(".convmerge-cache") / "llm.sqlite"


def _scale(value: str) -> tuple[int, int]:
    from convmerge.label import parse_scale

    try:
        return parse_scale(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _positive_float(value: str) -> float:
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not f > 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value}")
    return f


def _add_label(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "label",
        help="Label rows with a model's answer (one class, or a point on a scale) "
        "written to meta, through an OpenAI-compatible endpoint; cached",
        description="Each row's conversation is sent to the endpoint and the parsed answer "
        "is written to meta.NAME; no row is dropped. Only local and private hosts unless "
        "--allow-external: rows are sent as they are, so remove personal data before "
        "using an external API. Answers are cached, so a re-run sends nothing new.",
    )
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument("--endpoint", required=True, metavar="URL",
                   help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")  # fmt: skip
    p.add_argument("--model", required=True, help="Model name the server serves")
    p.add_argument("--name", required=True, help="Write the label to meta.NAME")
    task = p.add_mutually_exclusive_group(required=True)
    task.add_argument("--classes", metavar="A,B,...",
                      help="Comma-separated class names, at least 2")  # fmt: skip
    task.add_argument("--scale", type=_scale, metavar="LO-HI",
                      help="Integer scale, e.g. 1-5")  # fmt: skip
    p.add_argument("--question", default=None,
                   help="What to ask; needed with --classes, and with --scale "
                   "when there is no --rubric")  # fmt: skip
    p.add_argument("--rubric", type=Path, default=None, metavar="FILE",
                   help="Rubric text for --scale")  # fmt: skip
    p.add_argument("--prompt", type=Path, default=None, metavar="FILE",
                   help="Replace the built-in prompt: a template with $conversation, "
                   "$classes, $scale, $question, $rubric")  # fmt: skip
    p.add_argument("--meta-key", default="meta", metavar="KEY",
                   help="Write under this key instead of meta")  # fmt: skip
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing meta.NAME (otherwise the run refuses)")  # fmt: skip
    p.add_argument("--max-chars", type=_positive_int, default=8000, metavar="N",
                   help="Cut longer conversations from the middle (default 8000)")  # fmt: skip
    p.add_argument("--cache", type=Path, default=None, metavar="FILE",
                   help=f"SQLite answer cache (default: {DEFAULT_CACHE.as_posix()} next "
                   "to the output)")  # fmt: skip
    p.add_argument("--cache-only", action="store_true",
                   help="Make no calls: rows not in the cache get null")  # fmt: skip
    p.add_argument("--api-key-env", default=None, metavar="NAME",
                   help="Read the API key from this environment variable")  # fmt: skip
    p.add_argument("--allow-external", action="store_true",
                   help="Allow a host that is not local or private")  # fmt: skip
    p.add_argument("--workers", type=_positive_int, default=4, help="Concurrent requests (4)")
    p.add_argument("--timeout", type=_positive_float, default=60.0,
                   help="Seconds per request (default 60)")  # fmt: skip
    p.add_argument("--max-calls", type=_positive_int, default=None, metavar="N",
                   help="Stop after N network calls (cache hits are free); exit 1")  # fmt: skip


def _cmd_label(args: argparse.Namespace) -> None:
    from convmerge.convert import REPORT_VERSION
    from convmerge.label import LabelStats, label_jsonl
    from convmerge.llm import ExternalHostError, LLMError

    for f in (args.input, args.rubric, args.prompt):
        if f is not None and not f.is_file():
            print(f"error: input file not found: {f}", file=sys.stderr)
            sys.exit(1)
    classes = args.classes.split(",") if args.classes is not None else None
    cache = args.cache if args.cache is not None else args.output.parent / DEFAULT_CACHE
    stats = LabelStats()
    try:
        label_jsonl(
            args.input, args.output, endpoint=args.endpoint, model=args.model,
            name=args.name, classes=classes, scale=args.scale, question=args.question,
            rubric=args.rubric.read_text(encoding="utf-8") if args.rubric else None,
            prompt=args.prompt.read_text(encoding="utf-8") if args.prompt else None,
            meta_key=args.meta_key, overwrite=args.overwrite, max_chars=args.max_chars,
            cache=cache, cache_only=args.cache_only, api_key_env=args.api_key_env,
            allow_external=args.allow_external, workers=args.workers,
            timeout=args.timeout, max_calls=args.max_calls, stats=stats,
        )  # fmt: skip
    except (ValueError, ExternalHostError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        print(f"{args.output} was not written; {count(stats.calls, 'call')} made, answers "
              f"cached in {cache}", file=sys.stderr)  # fmt: skip
        sys.exit(1)
    print(json.dumps({"version": REPORT_VERSION, **stats.to_report()}, ensure_ascii=False,
                     indent=2))  # fmt: skip
    for n, what in ((stats.parse_failures, "answer(s) did not parse after a retry"),
                    (stats.call_failures, "row(s) got no answer from the endpoint"),
                    (stats.unreadable, "row(s) had no readable conversation"),
                    (stats.truncated, "conversation(s) were cut to --max-chars")):  # fmt: skip
        if n:
            print(f"warning: {n:,} {what}", file=sys.stderr)
    print(f"wrote {stats.rows:,} rows ({stats.labeled:,} labeled, {stats.calls:,} calls, "
          f"{stats.cache_hits:,} cached) -> {args.output}", file=sys.stderr)  # fmt: skip
