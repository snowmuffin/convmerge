"""inspect, normalize, dedupe, and turns commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from convmerge.cli._common import add_progress_flag as _add_progress_flag
from convmerge.cli._common import positive_int as _positive_int


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

    if not Path(args.input).is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
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


def _add_split(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "split",
        help="Split a JSONL file into train and validation sets by content hash",
    )
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, required=True, help="Train output")
    p.add_argument(
        "--val-output",
        type=Path,
        default=None,
        help="Validation output (default: <output stem>.val.jsonl next to --output)",
    )
    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument(
        "--val", type=_fraction, default=None, metavar="FRACTION",
        help="Send about this fraction of rows to validation (e.g. 0.05)",
    )  # fmt: skip
    size.add_argument(
        "--val-rows", type=_non_negative_int, default=None, metavar="N",
        help="Send exactly N rows to validation",
    )  # fmt: skip
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Hash only these top-level keys, so rows sharing them stay on one side",
    )


def _cmd_split(args: argparse.Namespace) -> None:
    from convmerge.split import SplitStats, default_val_path, split_jsonl

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    val_output = args.val_output or default_val_path(args.output)
    stats = SplitStats()
    split_jsonl(
        args.input, args.output, val_output,
        val=args.val, val_rows=args.val_rows, seed=args.seed, keys=args.keys, stats=stats,
    )  # fmt: skip
    print(
        f"train={stats.train:,} -> {args.output}\nval={stats.val:,} -> {val_output}",
        file=sys.stderr,
    )
    if stats.first_invalid_line is not None:
        print(
            f"warning: dropped {stats.invalid_json:,} invalid JSON lines "
            f"(first at line {stats.first_invalid_line})",
            file=sys.stderr,
        )


def _fraction(value: str) -> float:
    f = float(value)
    if not 0.0 < f < 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1 (exclusive), got {value}")
    return f


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return n


def _add_llamafactory_info(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "llamafactory-info",
        help="Write the LLaMA-Factory dataset_info.json entry for a converted file",
    )
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--name", required=True, help="Dataset name to register (dataset: NAME)")
    p.add_argument(
        "--info",
        type=Path,
        default=None,
        help="dataset_info.json to add the entry to (created if missing); "
        "without it the entry is printed",
    )
    p.add_argument(
        "--file-name",
        default=None,
        help="file_name to record (default: the input path relative to --info's "
        "directory, or its name)",
    )


def _cmd_llamafactory_info(args: argparse.Namespace) -> None:
    from convmerge.llamafactory import dataset_info_entry, relative_file_name, update_dataset_info

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    file_name = args.file_name
    if file_name is None and args.info is not None:
        file_name = relative_file_name(args.input, args.info)
    try:
        entry = dataset_info_entry(args.input, file_name=file_name)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    if args.info is None:
        print(json.dumps({args.name: entry}, ensure_ascii=False, indent=2))
        return
    changed = update_dataset_info(args.info, args.name, entry)
    state = "updated" if changed else "unchanged"
    print(f"{args.info}: {args.name!r} {state} (use dataset: {args.name})", file=sys.stderr)


def _add_tokens(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tokens",
        help="Token lengths and chat-template check for a model (optionally filter by length)",
        description="Render every row with the model's chat template, count tokens, and "
        "report rows the template rejects; -o keeps rows that render and fit --max-tokens. "
        'Needs pip install "convmerge[tokens]" (transformers; no PyTorch).',
    )
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument(
        "--tokenizer", required=True, help="Model name on the Hub or a local tokenizer directory"
    )
    p.add_argument("--revision", default=None, help="Tokenizer revision (branch, tag, commit)")
    p.add_argument("--hf-token", default=None, help="Token for gated tokenizers (or HF_TOKEN)")
    p.add_argument(
        "--chat-template", type=Path, default=None, metavar="JINJA",
        help="Use this chat template file instead of the tokenizer's",
    )  # fmt: skip
    p.add_argument("--max-tokens", type=_positive_int, default=None)
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Write rows that render and fit --max-tokens here",
    )  # fmt: skip
    p.add_argument("--rejects", type=Path, default=None, help="Write the other rows here")


def _cmd_tokens(args: argparse.Namespace) -> None:
    from convmerge.tokens import TokenStats, check_tokens, load_tokenizer

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    try:
        template = args.chat_template.read_text(encoding="utf-8") if args.chat_template else None
        tok = load_tokenizer(args.tokenizer, revision=args.revision, token=args.hf_token)
    except (ImportError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    stats = TokenStats()
    try:
        check_tokens(
            args.input, tokenizer=tok, max_tokens=args.max_tokens, output=args.output,
            rejects=args.rejects, chat_template=template, stats=stats,
        )  # fmt: skip
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    stats.tokenizer = args.tokenizer
    print(json.dumps(stats.to_report(), ensure_ascii=False, indent=2))
    failed = sum(stats.template_errors.values())
    if failed:
        top = max(stats.template_errors, key=stats.template_errors.__getitem__)
        print(f"warning: {failed:,} rows fail the chat template (most common: {top})",
              file=sys.stderr)  # fmt: skip
    if stats.over_limit:
        print(f"warning: {stats.over_limit:,} rows exceed {args.max_tokens:,} tokens",
              file=sys.stderr)  # fmt: skip
    if args.output is not None:
        print(f"kept {stats.kept:,} -> {args.output}; rejected {stats.rejected:,}",
              file=sys.stderr)  # fmt: skip
    elif failed or stats.over_limit:
        sys.exit(1)  # check mode: problems found


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
