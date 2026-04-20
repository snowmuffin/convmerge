"""Iterate a manifest and dispatch each entry to the right backend."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from convmerge.fetch.auth import AuthConfig, redact_url, resolve_token
from convmerge.fetch.manifest import (
    DatasetEntry,
    EntryKind,
    Manifest,
    classify_entry,
    sanitize_name,
)

LogFn = Callable[[str], None]


@dataclass
class FetchResult:
    """Summary returned by :func:`run_manifest`."""

    succeeded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.skipped) + len(self.failed)


def run_manifest(
    manifest: Manifest,
    *,
    output_root: str | Path | None = None,
    only: list[str] | None = None,
    hf_token: str | None = None,
    github_token: str | None = None,
    log: LogFn = print,
) -> FetchResult:
    """Execute every selected entry in ``manifest`` sequentially.

    ``output_root`` overrides the manifest default when provided. ``only``
    filters the entries by name. ``hf_token`` / ``github_token`` take highest
    priority over the manifest ``auth`` block and process env.
    """
    base_root = Path(output_root) if output_root else Path(manifest.defaults.output_root)
    base_root.mkdir(parents=True, exist_ok=True)

    hf_tok = resolve_token(manifest.auth.hf, explicit=hf_token)
    gh_tok = resolve_token(manifest.auth.github, explicit=github_token)

    entries = _select_entries(manifest.datasets, only)
    result = FetchResult()

    for entry in entries:
        dst = _entry_output_path(entry, base_root)
        try:
            kind = classify_entry(entry)
        except ValueError as e:
            _record_error(result, entry.name, str(e), on_error=manifest.defaults.on_error, log=log)
            continue

        if manifest.defaults.resume and _already_fetched(dst, kind):
            log(f"[skip] {entry.name} (already present at {dst})")
            result.skipped.append(entry.name)
            continue

        log(f"[fetch] {entry.name} ({kind}) -> {dst}")
        try:
            _dispatch(entry, kind, dst, hf_tok=hf_tok, gh_tok=gh_tok)
        except Exception as e:  # noqa: BLE001  (report, let on_error decide)
            detail = f"{type(e).__name__}: {e}"
            trace_tail = traceback.format_exc(limit=2).strip().splitlines()[-1:]
            full = detail + (f" | {trace_tail[0]}" if trace_tail else "")
            _record_error(result, entry.name, full, on_error=manifest.defaults.on_error, log=log)
            continue
        result.succeeded.append(entry.name)

    log(
        f"[done] ok={len(result.succeeded)} skipped={len(result.skipped)} "
        f"failed={len(result.failed)}"
    )
    return result


def _select_entries(
    datasets: tuple[DatasetEntry, ...], only: list[str] | None
) -> list[DatasetEntry]:
    if not only:
        return list(datasets)
    wanted = set(only)
    matched = [d for d in datasets if d.name in wanted]
    missing = wanted - {d.name for d in matched}
    if missing:
        raise ValueError(f"Unknown dataset name(s) in --only: {sorted(missing)}")
    return matched


def _entry_output_path(entry: DatasetEntry, base_root: Path) -> Path:
    if entry.output:
        return Path(entry.output)
    return base_root / sanitize_name(entry.name)


def _already_fetched(dst: Path, kind: EntryKind) -> bool:
    if kind in ("hf", "url_raw"):
        # These write a single file (.jsonl / raw). Treat as done if it exists
        # with non-zero size, or if the parent directory contains a non-empty file
        # (covers the ``output: dir`` override + default ``{name}.jsonl`` name).
        if dst.is_file() and dst.stat().st_size > 0:
            return True
        if dst.is_dir():
            return any(p.is_file() and p.stat().st_size > 0 for p in dst.iterdir())
        parent_file = dst.with_suffix(".jsonl")
        if parent_file.is_file() and parent_file.stat().st_size > 0:
            return True
        return False
    # Tree / clone produce a directory with one or more files.
    if not dst.is_dir():
        return False
    return any(dst.iterdir())


def _dispatch(
    entry: DatasetEntry,
    kind: EntryKind,
    dst: Path,
    *,
    hf_tok: str | None,
    gh_tok: str | None,
) -> None:
    if kind == "hf":
        _run_hf(entry, dst, token=hf_tok)
        return
    if kind == "url_raw":
        _run_raw(entry, dst, token=gh_tok)
        return
    if kind == "url_github_tree":
        _run_tree(entry, dst, token=gh_tok)
        return
    if kind == "url_github_clone":
        _run_clone(entry, dst, token=gh_tok)
        return
    raise AssertionError(f"Unhandled entry kind: {kind!r}")


def _run_hf(entry: DatasetEntry, dst: Path, *, token: str | None) -> None:
    from convmerge.fetch.hf import download_hf_dataset

    target = dst if dst.suffix else dst.with_suffix(".jsonl")
    download_hf_dataset(
        entry.hf or "",
        target,
        config=entry.config,
        split=entry.split,
        token=token,
    )


def _run_raw(entry: DatasetEntry, dst: Path, *, token: str | None) -> None:
    from convmerge.fetch.github import download_raw_file

    url = entry.url or ""
    suffix = _raw_suffix(url)
    target = dst if dst.suffix else dst.with_suffix(suffix)
    download_raw_file(url, target, token=token)


def _run_tree(entry: DatasetEntry, dst: Path, *, token: str | None) -> None:
    from convmerge.fetch.github import fetch_repo_tree_files

    fetch_repo_tree_files(entry.url or "", dst, ext=entry.ext, token=token)


def _run_clone(entry: DatasetEntry, dst: Path, *, token: str | None) -> None:
    from convmerge.fetch.git import clone_repo

    clone_repo(entry.url or "", dst, token=token, lfs=entry.lfs)


def _raw_suffix(url: str) -> str:
    lowered = url.lower()
    for s in (".json.gz", ".jsonl", ".json"):
        if lowered.endswith(s):
            return s
    return ".jsonl"


def _record_error(
    result: FetchResult,
    name: str,
    msg: str,
    *,
    on_error: str,
    log: LogFn,
) -> None:
    log(f"[fail] {name}: {msg}")
    result.failed.append((name, msg))
    if on_error == "fail":
        raise RuntimeError(f"Fetch failed for {name!r}: {msg}")


# Re-export for testing / external use.
__all__ = [
    "AuthConfig",
    "FetchResult",
    "redact_url",
    "run_manifest",
]
