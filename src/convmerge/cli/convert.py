"""convert, validate, formats, and preset commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from convmerge.cli._common import add_progress_flag as _add_progress_flag
from convmerge.cli._common import config_errors, encoding_advice
from convmerge.cli._common import positive_int as _positive_int
from convmerge.convert import REPORT_VERSION, ConvertStats, convert_file


def _add_convert(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "convert",
        help="Convert a JSONL file using a source adapter and output format",
        description="Convert a JSONL file using a source adapter and output format. "
        "YAML presets require convmerge[preset].",
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Input JSONL path")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output JSONL path")
    p.add_argument(
        "--preset",
        type=Path,
        default=None,
        help="YAML/JSON preset file (install convmerge[preset] for YAML)",
    )
    p.add_argument(
        "--from",
        dest="adapter",
        default=None,
        metavar="ADAPTER",
        help="Source adapter: alpaca, sharegpt, chat, auto (optional if --preset sets it)",
    )
    p.add_argument(
        "--format",
        "-f",
        dest="output_format",
        default=None,
        metavar="FORMAT",
        help="Output format: messages (default), alpaca, sharegpt, preference (DPO "
        "pairs), sharegpt-preference; see `convmerge formats`",
    )
    p.add_argument(
        "--adapter-kwargs",
        default=None,
        metavar="JSON",
        help=(
            'JSON object merged on the preset, e.g. {"chat":{"pairwise_mode":"both"}} '
            'or {"sharegpt":{"turn_mode":"full"}}'
        ),
    )
    p.add_argument(
        "--encoding", default="utf-8",
        help="Input file encoding, such as cp949 (default: utf-8); output is always UTF-8",
    )  # fmt: skip
    p.add_argument(
        "--preference",
        choices=("chosen", "rejected"),
        default=None,
        help="Preference (DPO / reward) data as SFT: train on the chosen (or rejected) "
        "answer. For DPO pairs use --format preference instead",
    )
    p.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Convert with N processes (default 1). Output order and stats are "
        "identical to a single-process run",
    )
    p.add_argument(
        "--on-invalid",
        choices=("drop", "keep", "fail"),
        default="drop",
        help="What to do with examples that fail validation (no user turn, empty "
        "messages, orphan tool results, ...): drop and count them (default), "
        "keep them, or stop with an error",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write a JSON report of counts, drop reasons, and sample line numbers",
    )
    p.add_argument(
        "--tool-arguments",
        choices=("string", "object"),
        default=None,
        help="messages format: write tool-call arguments as a JSON string "
        "(default, OpenAI style) or as a JSON object",
    )
    p.add_argument(
        "--keep-meta",
        nargs="?",
        const="*",
        default=None,
        metavar="KEYS",
        help="Also write provenance (source, id, branch) under 'meta'; "
        "optionally only these comma-separated keys, e.g. --keep-meta source,id",
    )
    p.add_argument("--meta-key", default=None, help="Output key for --keep-meta (default: meta)")
    p.add_argument(
        "--meta",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Also write this constant field under 'meta' on every row (repeatable), "
        "e.g. --meta dataset=kullm --meta license=apache-2.0",
    )
    p.add_argument(
        "--alpaca-multiturn",
        choices=("flatten", "history", "drop"),
        default=None,
        help="alpaca format, conversations longer than one pair: flatten into one "
        "instruction (default, lossy), write a LLaMA-Factory 'history' list, or drop",
    )
    p.add_argument(
        "--reasoning",
        choices=("keep", "inline", "reasoning_content", "thinking", "drop"),
        default=None,
        help="Assistant reasoning traces: keep them as found (default), write them "
        "inline as <think>...</think>, move them to a reasoning_content (Qwen3, "
        "DeepSeek) or thinking (gpt-oss) field, or drop them",
    )
    p.add_argument(
        "--tool-content",
        choices=("empty", "null"),
        default=None,
        help='content of assistant turns that only call tools: "" (default; every '
        "common chat template accepts it) or null",
    )
    p.add_argument(
        "--train-turns",
        choices=("all", "last", "data"),
        default=None,
        help='last: mark every assistant turn but the last with "train": false; '
        "data: write the flag the dataset gives each assistant turn (train / loss / "
        "weight keys, metadata.train_turns) (messages format; axolotl "
        "message_field_training: train)",
    )
    g = p.add_argument_group("fixes for strict chat templates (all off by default)")
    g.add_argument(
        "--system",
        dest="system_mode",
        choices=("keep", "fold", "drop"),
        default=None,
        help="fold: move system turns into the first user turn (templates without a "
        "system role, e.g. Gemma 2); drop: remove them",
    )
    g.add_argument(
        "--merge-consecutive",
        action="store_const",
        const=True,
        default=None,
        help="Join consecutive user (or assistant) turns, for templates that require "
        "alternating roles",
    )
    g.add_argument(
        "--split-turns",
        action="store_const",
        const=True,
        default=None,
        help="One example per user turn; earlier turns keep their answers but lose "
        "their reasoning (multi-turn reasoning data for Qwen3-style templates)",
    )
    g.add_argument(
        "--reasoning-turns",
        choices=("all", "last"),
        default=None,
        help="last: remove reasoning from assistant turns before the last user turn "
        "(what Qwen3 / gpt-oss templates render)",
    )
    g.add_argument(
        "--leading-assistant",
        choices=("keep", "drop"),
        default=None,
        help="drop: remove assistant turns before the first user turn (answers to a "
        "withheld prompt, which are otherwise dropped as withheld_prompt, or a greeting) "
        "and tool results right after it that answer no call (orphan_tool_message)",
    )
    _add_progress_flag(p)


def _cmd_convert(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    from convmerge.config import build_convert_config

    try:
        cfg = build_convert_config(
            preset_path=args.preset,
            adapter=args.adapter,
            output_format=args.output_format,
            encoding=args.encoding,
            adapter_kwargs_json=args.adapter_kwargs,
            emit_overrides=_emit_overrides(args),
            preference=args.preference,
            transform_overrides=_transform_overrides(args),
        )
    except config_errors() as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    from convmerge.convert import InvalidExampleError
    from convmerge.io import bad_bytes_seen
    from convmerge.progress import progress_enabled

    stats = ConvertStats()
    bad_bytes_before = bad_bytes_seen()
    try:
        n_in, n_out = convert_file(
            args.input,
            args.output,
            adapter_name=cfg.adapter,
            output_format=cfg.output_format,
            encoding=cfg.encoding,
            adapter_options=cfg.adapter_options,
            progress=progress_enabled(args.progress),
            stats=stats,
            on_invalid=args.on_invalid,
            emit_options=cfg.emit_options,
            workers=max(1, args.workers),
            transform_options=cfg.transform_options,
        )
    except InvalidExampleError as e:
        print(f"error: {e} (use --on-invalid drop or keep to continue)", file=sys.stderr)
        sys.exit(1)
    print(f"read {n_in} lines, wrote {n_out} examples", file=sys.stderr)
    if stats.grouped:
        print(
            f"read OpenAssistant message rows as trees: {stats.grouped:,} rows joined "
            "the conversation of an earlier row",
            file=sys.stderr,
        )
    advice = encoding_advice(bad_bytes_before, cfg.encoding, has_encoding_flag=True)
    if n_out == 0 and not advice:  # a bad encoding is explained with the skipped lines below
        for hint in _explain_empty(args.input, cfg.adapter, cfg.encoding, stats):
            print(f"hint: {hint}", file=sys.stderr)
    if stats.reasoning:
        print(f"{stats.reasoning:,} examples carry a reasoning trace", file=sys.stderr)
    if stats.transforms:
        fixes = ", ".join(f"{k}={n:,}" for k, n in sorted(stats.transforms.items()))
        print(f"fixes: {fixes}", file=sys.stderr)
    _print_drop_summary(stats, kept=args.on_invalid == "keep")
    if args.report:
        _write_report(args.report, stats)
    if stats.skipped:
        print(
            f"warning: skipped {stats.skipped:,} lines "
            f"(invalid JSON={stats.invalid_json:,}, non-object={stats.non_object:,}, "
            f"no example from adapter={stats.no_example:,})",
            file=sys.stderr,
        )
    if stats.first_invalid_line is not None:
        print(
            f"warning: first invalid JSON at line {stats.first_invalid_line}; "
            f"{advice or 'run `convmerge normalize` first to repair the file'}",
            file=sys.stderr,
        )


# Rows sampled from the head of the file to explain an empty result.
_EXPLAIN_ROWS = 50


def _explain_empty(path: Path, adapter: str, encoding: str, stats: ConvertStats) -> list[str]:
    """Why nothing was written, and what to run instead (best effort)."""
    from convmerge.adapters.chat import iter_from_chat_line
    from convmerge.io import iter_jsonl
    from convmerge.normalize.jsonl import detect_jsonl_shape
    from convmerge.validate import validate_example

    if stats.lines_read == stats.blank:
        return []
    if stats.invalid_json or stats.non_object:
        try:
            shape = detect_jsonl_shape(path)
        except (OSError, ValueError, RecursionError):
            shape = "invalid"
        if shape in ("json_array", "single_line", "jsonl_of_arrays"):
            return [
                f"the file is not one JSON object per line (looks like {shape}); "
                f"run `convmerge normalize -i {path} -o <out.jsonl>` first"
            ]
    missing = {"no_messages", "no_user", "no_assistant"}
    if not stats.no_example and not missing & set(stats.drop_reasons):
        return []
    if "preference_record" in stats.drop_reasons:
        return []  # the drop summary says how to read preference records
    rows = []
    for line in iter_jsonl(path, encoding=encoding):
        if isinstance(line.value, dict):
            rows.append(line.value)
            if len(rows) >= _EXPLAIN_ROWS:
                break
    if not rows:
        return []
    if adapter not in ("auto", "chat"):
        ok = sum(
            1 for row in rows if any(not validate_example(ex) for ex in iter_from_chat_line(row))
        )
        if ok:
            return [f"{ok} of the first {len(rows)} rows convert with --from auto"]
    keys = ", ".join(list(rows[0])[:12])
    return [
        f"no conversation fields recognised; the first row has keys: {keys}",
        "map them with --from map --adapter-kwargs "
        '\'{"map": {"user": "<field>", "assistant": "<field>"}}\' '
        "(`convmerge inspect` lists every field)",
    ]


def _emit_overrides(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    if args.tool_arguments is not None:
        out["tool_arguments"] = args.tool_arguments
    if args.keep_meta is not None:
        keys = [k.strip() for k in args.keep_meta.split(",") if k.strip()]
        out["keep_meta"] = True if args.keep_meta == "*" else tuple(keys)
    if args.meta_key is not None:
        out["meta_key"] = args.meta_key
    if args.meta:
        values: dict[str, str] = {}
        for item in args.meta:
            key, sep, value = item.partition("=")
            if not sep or not key.strip():
                raise ValueError(f"--meta expects KEY=VALUE, got {item!r}")
            values[key.strip()] = value
        out["meta_values"] = values
    if args.alpaca_multiturn is not None:
        out["alpaca_multiturn"] = args.alpaca_multiturn
    if args.reasoning is not None:
        out["reasoning"] = args.reasoning
    if args.tool_content is not None:
        out["tool_content"] = args.tool_content
    if args.train_turns is not None:
        out["train_turns"] = args.train_turns
    return out


def _transform_overrides(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    if args.system_mode is not None:
        out["system"] = args.system_mode
    if args.merge_consecutive is not None:
        out["merge_consecutive"] = True
    if args.split_turns is not None:
        out["split_turns"] = True
    if args.reasoning_turns is not None:
        out["reasoning_turns"] = args.reasoning_turns
    if args.leading_assistant is not None:
        out["leading_assistant"] = args.leading_assistant
    return out


def _print_drop_summary(stats: ConvertStats, *, kept: bool = False) -> None:
    _print_lossy_summary(stats)
    if not stats.drop_reasons:
        return
    reasons = ", ".join(f"{r}={n:,}" for r, n in sorted(stats.drop_reasons.items()))
    if kept:
        print(f"warning: kept {stats.kept_invalid:,} invalid examples ({reasons})", file=sys.stderr)
    if stats.dropped:
        print(f"warning: dropped {stats.dropped:,} examples ({reasons})", file=sys.stderr)
    hints = [_DROP_HINTS[r] for r in sorted(stats.drop_reasons) if r in _DROP_HINTS]
    for hint in hints:
        print(f"hint: {hint}", file=sys.stderr)


_DROP_HINTS = {
    "preference_record": (
        "preference_record: these are chosen/rejected pairs; use --format preference "
        "for DPO data, or --preference chosen to train on the chosen answers"
    ),
    "unrepresentable_not_preference": (
        "unrepresentable_not_preference: --format preference writes only records "
        "that have both a chosen and a rejected answer"
    ),
    "map_path_missing": (
        "map_path_missing: a --from map path matched nothing in these records; check the "
        "paths against a sample row (a.b for keys, a[0] for an item, a[] for every item)"
    ),
    "withheld_prompt": (
        "withheld_prompt: the dataset left these prompts null; --leading-assistant drop "
        "removes the answer to the missing prompt and keeps the rest of the conversation"
    ),
    "unrepresentable_role_order": (
        "unrepresentable_role_order: --merge-consecutive joins repeated user or "
        "assistant turns; --system fold moves a system turn into the first user turn"
    ),
}


def _print_lossy_summary(stats: ConvertStats) -> None:
    for reason, n in sorted(stats.lossy.items()):
        hint = (
            " (use --alpaca-multiturn history to keep turns, or drop)"
            if reason == "lossy_multiturn_flattened"
            else ""
        )
        print(f"warning: {n:,} examples written lossily: {reason}{hint}", file=sys.stderr)


def _write_report(path: Path, stats: ConvertStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": REPORT_VERSION, **stats.to_report()}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"report: {path}", file=sys.stderr)


def _add_validate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "validate",
        help="Check a JSONL file's examples for SFT problems (exit 1 if any are invalid)",
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Input JSONL path")
    p.add_argument(
        "--from",
        dest="adapter",
        default="chat",
        metavar="ADAPTER",
        help="Adapter used to read records (default: chat, which reads messages rows)",
    )
    p.add_argument("--encoding", default="utf-8", help="Input file encoding (default: utf-8)")


def _cmd_validate(args: argparse.Namespace) -> None:
    from convmerge.convert import validate_file

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    try:
        stats = validate_file(args.input, adapter_name=args.adapter, encoding=args.encoding)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    report: dict[str, object] = {"version": REPORT_VERSION, **stats.to_report()}
    report["valid"] = report.pop("written")
    report["invalid"] = report.pop("dropped")
    for key in ("kept_invalid",):
        report.pop(key)
    from convmerge.arrow import type_conflicts

    conflicts = type_conflicts(args.input, encoding=args.encoding)
    if conflicts:
        report["type_conflicts"] = [c.to_report() for c in conflicts]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if conflicts:
        print(
            f"warning: {len(conflicts)} field(s) change type between rows; "
            'datasets.load_dataset("json") fails on them before datasets 4.8 '
            "(LLaMA-Factory installs 4.0) with ArrowInvalid",
            file=sys.stderr,
        )
        for c in conflicts[:5]:
            kinds = ", ".join(f"{k} (line {n})" for k, n in c.first_line.items())
            print(f"  {c.path}: {kinds}; {c.hint()}", file=sys.stderr)
    if stats.dropped or stats.skipped:
        sys.exit(1)


def _add_formats(sub: argparse._SubParsersAction) -> None:
    sub.add_parser(
        "formats",
        help="List source adapters (--from) and output formats (--format), including plugins",
    )


def _cmd_formats(args: argparse.Namespace) -> None:
    from convmerge.adapters import BUILTIN_ADAPTERS, available_adapters
    from convmerge.emitters import BUILTIN_FORMATS, available_formats

    def fmt(names: list[str], builtin: frozenset[str]) -> str:
        return ", ".join(n if n in builtin else f"{n} (plugin)" for n in names)

    print(f"adapters (--from):  {fmt(available_adapters(), BUILTIN_ADAPTERS)}")
    print(f"formats (--format): {fmt(available_formats(), BUILTIN_FORMATS)}")


def _add_preset(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "preset",
        help="Create or validate convert preset files (install convmerge[preset] for YAML)",
        description="Create or validate convert preset files. YAML requires convmerge[preset].",
    )
    subp = p.add_subparsers(dest="preset_action", required=True)
    pi = subp.add_parser("init", help="Write a commented YAML template")
    pi.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file (default: print to stdout)",
    )
    pv = subp.add_parser("validate", help="Validate a preset YAML/JSON file")
    pv.add_argument("path", type=Path)


def _cmd_preset_init(args: argparse.Namespace) -> None:
    from convmerge.preset import PRESET_TEMPLATE_YAML

    if args.output:
        args.output.write_text(PRESET_TEMPLATE_YAML, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(PRESET_TEMPLATE_YAML, end="")


def _cmd_preset_validate(args: argparse.Namespace) -> None:
    from convmerge.preset import validate_preset_file

    try:
        validate_preset_file(args.path)
    except config_errors() as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print("ok", file=sys.stderr)


def _cmd_preset(args: argparse.Namespace) -> None:
    if args.preset_action == "init":
        _cmd_preset_init(args)
    else:
        _cmd_preset_validate(args)
