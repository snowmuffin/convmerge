"""inspect, normalize, dedupe, and turns commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from convmerge.cli._common import add_progress_flag as _add_progress_flag


def _add_inspect(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "inspect",
        help=(
            "Profile the schema/structure of a JSON/JSONL file "
            "(keys, value types, nesting, presence, sample values)"
        ),
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Input .json or .jsonl path")
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Sample only the first N records (recommended for large files)",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="Sample values to show per field (default: 3)",
    )


def _cmd_inspect(args: argparse.Namespace) -> None:
    from convmerge.normalize.schema import profile_schema

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    try:
        report = profile_schema(
            args.input,
            max_rows=args.max_rows,
            max_examples=args.max_examples,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _add_normalize(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "normalize",
        help=(
            "Normalize parquet/json/jsonl files in a directory into clean JSONL "
            "(install convmerge[parquet] for .parquet inputs)"
        ),
        description="Normalize parquet/json/jsonl files in a directory into clean JSONL. "
        "Parquet inputs require convmerge[parquet].",
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Input file or directory")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output file (when --input is a file) or directory",
    )
    p.add_argument(
        "--array-key",
        default="conversation",
        help="Key that wraps records which are JSON arrays, e.g. one conversation "
        "per line as a list of turns (default: conversation)",
    )


def _cmd_normalize(args: argparse.Namespace) -> None:
    from convmerge.normalize.files import normalize_path

    src: Path = args.input
    dst: Path = args.output
    if not src.exists():
        print(f"error: input not found: {src}", file=sys.stderr)
        sys.exit(1)
    if src.is_file():
        n = normalize_path(src, dst, array_key=args.array_key).records
        print(f"{src} -> {dst}: {n} records", file=sys.stderr)
        return

    def report(in_path: Path, out_path: Path, n: int | None, error: str | None) -> None:
        if error is not None:
            print(f"[fail] {in_path}: {error}", file=sys.stderr)
        else:
            print(f"[ok] {in_path} -> {out_path} ({n} records)", file=sys.stderr)

    result = normalize_path(src, dst, array_key=args.array_key, on_file=report)
    print(f"[done] {len(result.files)} files, {result.records} records", file=sys.stderr)


def _add_dedupe(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("dedupe", help="Remove duplicate rows from a JSONL file")
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Only hash these top-level keys (defaults to the whole record)",
    )
    p.add_argument("--algorithm", default="md5", choices=("md5", "sha256"))
    p.add_argument(
        "--seen-store",
        choices=("memory", "sqlite"),
        default="memory",
        help="Duplicate tracking: 'memory' (default, fastest) or 'sqlite' "
        "(disk-backed, bounded memory for huge unique-row counts)",
    )
    p.add_argument(
        "--seen-db",
        type=Path,
        default=None,
        help="sqlite store path (default: a temp file removed on completion)",
    )
    _add_progress_flag(p)


def _cmd_dedupe(args: argparse.Namespace) -> None:
    from convmerge.normalize.dedup import DedupeStats, deduplicate_jsonl
    from convmerge.progress import progress_enabled

    stats = DedupeStats()
    total, kept = deduplicate_jsonl(
        args.input,
        args.output,
        keys=args.keys,
        algorithm=args.algorithm,
        progress=progress_enabled(args.progress),
        seen_store=args.seen_store,
        seen_db=args.seen_db,
        stats=stats,
    )
    removed = total - kept
    pct = (removed / total * 100) if total else 0.0
    print(
        f"total={total:,} kept={kept:,} removed={removed:,} ({pct:.2f}%) "
        f"[duplicates={stats.duplicates:,} invalid_json={stats.invalid_json:,}]",
        file=sys.stderr,
    )
    if stats.first_invalid_line is not None:
        print(
            f"warning: dropped {stats.invalid_json:,} invalid JSON lines "
            f"(first at line {stats.first_invalid_line}); "
            "run `convmerge normalize` first to repair the file",
            file=sys.stderr,
        )


def _add_turns(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "turns",
        help="Analyze turn distribution of a messages-style JSONL file, optionally splitting it",
    )
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--single-out", type=Path, default=None)
    p.add_argument("--multi-out", type=Path, default=None)


def _cmd_turns(args: argparse.Namespace) -> None:
    from convmerge.normalize.turns import analyze_turn_distribution, split_by_turns

    if bool(args.single_out) != bool(args.multi_out):
        print("error: --single-out and --multi-out must be given together", file=sys.stderr)
        sys.exit(2)
    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    report = analyze_turn_distribution(args.input)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.single_out and args.multi_out:
        s, m = split_by_turns(
            args.input,
            single_out=args.single_out,
            multi_out=args.multi_out,
        )
        print(f"split: single={s:,} -> {args.single_out}", file=sys.stderr)
        print(f"split: multi ={m:,} -> {args.multi_out}", file=sys.stderr)
