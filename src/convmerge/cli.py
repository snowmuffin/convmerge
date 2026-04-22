"""CLI entrypoint for convmerge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from convmerge import __version__
from convmerge.convert import convert_file

FETCH_FILE_EXTENSIONS = (".parquet", ".json", ".jsonl")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert":
        _cmd_convert(args)
    elif args.command == "preset":
        if args.preset_action == "init":
            _cmd_preset_init(args)
        else:
            _cmd_preset_validate(args)
    elif args.command == "normalize":
        _cmd_normalize(args)
    elif args.command == "dedupe":
        _cmd_dedupe(args)
    elif args.command == "turns":
        _cmd_turns(args)
    elif args.command == "fetch":
        _cmd_fetch(args)
    else:  # pragma: no cover - argparse enforces ``required=True``
        parser.error(f"unknown command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convmerge",
        description=(
            "Fetch, normalize, and convert heterogeneous chat/instruct datasets "
            "into a single LLM training format."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_convert(subparsers)
    _add_normalize(subparsers)
    _add_dedupe(subparsers)
    _add_turns(subparsers)
    _add_fetch(subparsers)
    _add_preset(subparsers)

    return parser


def _add_convert(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "convert",
        help="Convert a JSONL file using a source adapter and output format",
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
        help="Output format: messages, alpaca (optional if --preset sets it)",
    )
    p.add_argument(
        "--adapter-kwargs",
        default=None,
        metavar="JSON",
        help='JSON object merged on the preset, e.g. {"chat":{"pairwise_mode":"both"}}',
    )
    p.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")


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
        )
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    n_in, n_out = convert_file(
        args.input,
        args.output,
        adapter_name=cfg.adapter,
        output_format=cfg.output_format,
        encoding=cfg.encoding,
        adapter_options=cfg.adapter_options,
    )
    print(f"read {n_in} lines, wrote {n_out} examples", file=sys.stderr)


def _add_preset(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("preset", help="Create or validate convert preset files")
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
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print("ok", file=sys.stderr)


def _add_normalize(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "normalize",
        help="Normalize parquet/json/jsonl files in a directory into clean JSONL",
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Input file or directory")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output file (when --input is a file) or directory",
    )


def _cmd_normalize(args: argparse.Namespace) -> None:
    src: Path = args.input
    dst: Path = args.output

    if src.is_file():
        n = _normalize_one_file(src, dst)
        print(f"{src} -> {dst}: {n} records", file=sys.stderr)
        return

    if not src.is_dir():
        print(f"error: input not found: {src}", file=sys.stderr)
        sys.exit(1)

    total_files = 0
    total_rows = 0
    for in_path in sorted(src.rglob("*")):
        if not in_path.is_file():
            continue
        if in_path.suffix.lower() not in FETCH_FILE_EXTENSIONS:
            continue
        rel = in_path.relative_to(src).with_suffix(".jsonl")
        out_path = dst / rel
        try:
            n = _normalize_one_file(in_path, out_path)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {in_path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        total_files += 1
        total_rows += n
        print(f"[ok] {in_path} -> {out_path} ({n} records)", file=sys.stderr)
    print(f"[done] {total_files} files, {total_rows} records", file=sys.stderr)


def _normalize_one_file(src: Path, dst: Path) -> int:
    # Imported lazily so that ``convmerge convert`` works without the
    # ``parquet`` extra when no parquet files are touched.
    from convmerge.normalize.jsonl import normalize_to_jsonl

    suffix = src.suffix.lower()
    if suffix == ".parquet":
        from convmerge.normalize.parquet import parquet_to_jsonl

        dst.parent.mkdir(parents=True, exist_ok=True)
        return parquet_to_jsonl(src, dst)
    return normalize_to_jsonl(src, dst)


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


def _cmd_dedupe(args: argparse.Namespace) -> None:
    from convmerge.normalize.dedup import deduplicate_jsonl

    total, kept = deduplicate_jsonl(
        args.input,
        args.output,
        keys=args.keys,
        algorithm=args.algorithm,
    )
    removed = total - kept
    pct = (removed / total * 100) if total else 0.0
    print(
        f"total={total:,} kept={kept:,} removed={removed:,} ({pct:.2f}%)",
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
    elif bool(args.single_out) != bool(args.multi_out):
        print(
            "error: --single-out and --multi-out must be given together",
            file=sys.stderr,
        )
        sys.exit(2)


def _add_fetch(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "fetch",
        help=(
            "Fetch training data via a YAML manifest, or a single "
            "hf://org/dataset / GitHub URL shortcut"
        ),
    )
    p.add_argument(
        "source",
        help="Path to a manifest YAML, an 'hf://org/dataset' URI, or a GitHub URL",
    )
    p.add_argument("--output", "-o", type=Path, default=None, help="Output root directory")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--github-token", default=None)
    p.add_argument("--only", nargs="+", default=None, help="Manifest mode: fetch only these names")
    p.add_argument(
        "--on-error",
        choices=("continue", "fail"),
        default=None,
        help="Override manifest defaults.on_error",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-download even when the output already exists",
    )
    # Shortcut-only flags (ignored in manifest mode)
    p.add_argument("--ext", nargs="+", default=None, help="GitHub URL mode: extension filter")
    p.add_argument("--mode", choices=("tree", "clone"), default=None)
    p.add_argument("--lfs", action="store_true")
    p.add_argument("--split", default=None, help="hf:// shortcut: dataset split")
    p.add_argument("--config", default=None, help="hf:// shortcut: dataset config")


def _cmd_fetch(args: argparse.Namespace) -> None:
    source = args.source

    if source.startswith("hf://") or source.startswith("https://") or source.startswith("http://"):
        _cmd_fetch_shortcut(args, source)
        return

    manifest_path = Path(source)
    if not manifest_path.is_file():
        print(
            f"error: manifest file not found: {manifest_path}\n"
            "Pass either a YAML manifest path, an hf://org/dataset URI, or a GitHub URL.",
            file=sys.stderr,
        )
        sys.exit(1)

    from convmerge.fetch.manifest import Defaults, load_manifest
    from convmerge.fetch.runner import run_manifest

    manifest = load_manifest(manifest_path)
    if args.on_error is not None or args.no_resume:
        manifest = _with_overridden_defaults(
            manifest,
            on_error=args.on_error,
            resume=False if args.no_resume else None,
        )

    result = run_manifest(
        manifest,
        output_root=args.output,
        only=args.only,
        hf_token=args.hf_token,
        github_token=args.github_token,
    )
    # Propagate failure when requested.
    if manifest.defaults.on_error == "fail" and result.failed:
        sys.exit(1)

    # Defaults reference for type checker.
    _ = Defaults


def _cmd_fetch_shortcut(args: argparse.Namespace, source: str) -> None:
    out_root = args.output or Path("./raw")
    out_root.mkdir(parents=True, exist_ok=True)

    if source.startswith("hf://"):
        from convmerge.fetch.hf import download_hf_dataset
        from convmerge.fetch.manifest import sanitize_name

        dataset_id = source[len("hf://") :]
        dst = out_root / f"{sanitize_name(dataset_id)}.jsonl"
        download_hf_dataset(
            dataset_id,
            dst,
            config=args.config,
            split=args.split,
            token=args.hf_token,
        )
        print(f"[ok] {dataset_id} -> {dst}", file=sys.stderr)
        return

    # http(s):// shortcuts
    from convmerge.fetch.git import clone_repo
    from convmerge.fetch.github import download_raw_file, fetch_repo_tree_files
    from convmerge.fetch.manifest import sanitize_name

    lowered = source.lower()
    name = sanitize_name(lowered.rstrip("/").rsplit("/", 1)[-1] or "fetch")

    if "raw.githubusercontent.com" in lowered or lowered.endswith((".json", ".jsonl", ".json.gz")):
        suffix = ".jsonl"
        for s in (".json.gz", ".jsonl", ".json"):
            if lowered.endswith(s):
                suffix = s
                break
        dst = out_root / f"{name}{suffix}"
        download_raw_file(source, dst, token=args.github_token)
        print(f"[ok] {source} -> {dst}", file=sys.stderr)
        return

    if "github.com" in lowered:
        dst = out_root / name
        if args.mode == "clone":
            clone_repo(source, dst, token=args.github_token, lfs=args.lfs)
        else:
            fetch_repo_tree_files(
                source,
                dst,
                ext=tuple(args.ext or ()),
                token=args.github_token,
            )
        print(f"[ok] {source} -> {dst}", file=sys.stderr)
        return

    print(
        f"error: unsupported URL: {source!r}. "
        "Only hf://, raw.githubusercontent.com, and github.com are supported.",
        file=sys.stderr,
    )
    sys.exit(1)


def _with_overridden_defaults(manifest, *, on_error, resume):
    from dataclasses import replace

    new_defaults = manifest.defaults
    if on_error is not None:
        new_defaults = replace(new_defaults, on_error=on_error)
    if resume is not None:
        new_defaults = replace(new_defaults, resume=resume)
    return replace(manifest, defaults=new_defaults)


if __name__ == "__main__":
    main()
