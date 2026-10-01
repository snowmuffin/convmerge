"""fetch command: YAML manifests and hf:// / GitHub URL shortcuts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from convmerge.cli._common import config_errors
from convmerge.cli._common import positive_int as _positive_int


def _add_fetch(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "fetch",
        help=(
            "Fetch training data via a YAML manifest, or a single "
            "hf://org/dataset / GitHub URL shortcut "
            "(convmerge[fetch] for YAML; [fetch-all], [fetch-hf], or [all] for HF entries)"
        ),
        description="Fetch training data via a YAML manifest, or a single "
        "hf://org/dataset / GitHub URL shortcut. "
        "Use convmerge[fetch] for YAML, [fetch-all]/[fetch-hf] for HF, or [all].",
    )
    p.add_argument(
        "source",
        help="Path to a manifest YAML, an 'hf://org/dataset' URI, or a GitHub URL",
    )
    p.add_argument("--output", "-o", type=Path, default=None, help="Output root directory")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--github-token", default=None)
    p.add_argument(
        "--only",
        nargs="+",
        action="extend",
        default=None,
        help="Manifest mode: fetch only these names",
    )
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
    p.add_argument(
        "--max-rows",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Fetch only the first N rows: HF streams them, raw/tree line files stop "
        "after N lines, LFS files are resolved without cloning (not for mode: clone)",
    )
    # Shortcut-only flags (ignored in manifest mode)
    p.add_argument(
        "--ext", nargs="+", action="extend", default=None, help="GitHub URL mode: extension filter"
    )
    p.add_argument("--mode", choices=("tree", "clone"), default=None)
    p.add_argument("--lfs", action="store_true")
    p.add_argument("--split", default=None, help="hf:// shortcut: dataset split")
    p.add_argument("--config", default=None, help="hf:// shortcut: dataset config")
    p.add_argument(
        "--revision", default=None,
        help="hf:// shortcut: commit sha, tag, or branch to pin the dataset to",
    )  # fmt: skip


def _cmd_fetch(args: argparse.Namespace) -> None:
    source = args.source

    if source.startswith("hf://") or source.startswith("https://") or source.startswith("http://"):
        try:
            _cmd_fetch_shortcut(args, source)
        except ImportError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        except Exception as e:  # noqa: BLE001  (network, HTTP, and Hub errors: one line, not a trace)
            print(f"error: fetch failed: {_one_line(e)}", file=sys.stderr)
            sys.exit(1)
        return

    manifest_path = Path(source)
    if not manifest_path.is_file():
        print(
            f"error: manifest file not found: {manifest_path}\n"
            "Pass either a YAML manifest path, an hf://org/dataset URI, or a GitHub URL.",
            file=sys.stderr,
        )
        sys.exit(2)

    from convmerge.fetch.manifest import Defaults, load_manifest
    from convmerge.fetch.runner import run_manifest

    try:
        manifest = load_manifest(manifest_path)
    except config_errors() as e:
        print(f"error: {manifest_path}: {e}", file=sys.stderr)
        sys.exit(2)
    if args.on_error is not None or args.no_resume:
        manifest = _with_overridden_defaults(
            manifest,
            on_error=args.on_error,
            resume=False if args.no_resume else None,
        )

    try:
        result = run_manifest(
            manifest,
            output_root=args.output,
            only=args.only,
            hf_token=args.hf_token,
            github_token=args.github_token,
            max_rows=args.max_rows,
        )
    except RuntimeError as e:
        if not str(e).startswith("Fetch failed for "):
            raise
        # An entry failed with on_error: fail; the runner already logged it.
        print(f"error: {_one_line(e)}", file=sys.stderr)
        sys.exit(1)
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
            max_rows=args.max_rows,
            revision=args.revision,
        )
        print(f"[ok] {dataset_id} -> {dst}", file=sys.stderr)
        return

    # http(s):// shortcuts
    from convmerge.fetch.auth import redact_url
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
        # The URL's file name already ends with the suffix; don't add it twice.
        stem = name[: -len(suffix)] if name.endswith(suffix) else name
        dst = out_root / f"{stem or 'fetch'}{suffix}"
        rows = None if suffix in (".json", ".json.gz") else args.max_rows
        download_raw_file(source, dst, token=args.github_token, max_rows=rows)
        print(f"[ok] {redact_url(source)} -> {dst}", file=sys.stderr)
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
                max_rows=args.max_rows,
            )
        print(f"[ok] {redact_url(source)} -> {dst}", file=sys.stderr)
        return

    print(
        f"error: unsupported URL: {redact_url(source)!r}. "
        "Only hf://, raw.githubusercontent.com, and github.com are supported.",
        file=sys.stderr,
    )
    sys.exit(2)


def _one_line(e: BaseException) -> str:
    """``Type: first line of the message``, with credentials removed from URLs."""
    from convmerge.fetch.auth import redact_url

    lines = str(e).strip().splitlines()
    return redact_url(f"{type(e).__name__}: {lines[0] if lines else ''}".rstrip(": "))


def _with_overridden_defaults(manifest, *, on_error, resume):
    from dataclasses import replace

    new_defaults = manifest.defaults
    if on_error is not None:
        new_defaults = replace(new_defaults, on_error=on_error)
    if resume is not None:
        new_defaults = replace(new_defaults, resume=resume)
    return replace(manifest, defaults=new_defaults)
