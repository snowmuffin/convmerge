"""Iterate a manifest and dispatch each entry to the right backend."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
_COMPLETION_MARKER_VERSION = 1


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

        if manifest.defaults.resume and _already_fetched(dst, kind, entry):
            log(f"[skip] {entry.name} (already present at {dst})")
            result.skipped.append(entry.name)
            continue

        log(f"[fetch] {entry.name} ({kind}) -> {dst}")
        try:
            output = _dispatch(entry, kind, dst, hf_tok=hf_tok, gh_tok=gh_tok)
        except Exception as e:  # noqa: BLE001  (report, let on_error decide)
            detail = f"{type(e).__name__}: {e}"
            trace_tail = traceback.format_exc(limit=2).strip().splitlines()[-1:]
            full = detail + (f" | {trace_tail[0]}" if trace_tail else "")
            _record_error(result, entry.name, full, on_error=manifest.defaults.on_error, log=log)
            continue
        try:
            _write_completion_marker(output)
        except OSError as e:
            # The download itself succeeded.  A marker failure only means the
            # next resume will conservatively fetch again.
            log(f"[warn] {entry.name}: could not write completion marker: {e}")
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


def _output_path(entry: DatasetEntry, kind: EntryKind, dst: Path) -> Path:
    if kind == "hf":
        return dst if dst.suffix else dst.with_suffix(".jsonl")
    if kind == "url_raw":
        return dst if dst.suffix else dst.with_suffix(_raw_suffix(entry.url or ""))
    return dst


def _completion_marker_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.fetch.json")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completion_snapshot(output: Path) -> dict[str, object] | None:
    if output.is_file():
        return {"type": "file", "size": output.stat().st_size, "sha256": _file_digest(output)}
    if not output.is_dir():
        return None
    files: list[dict[str, object]] = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        files.append(
            {
                "path": str(path.relative_to(output)),
                "size": path.stat().st_size,
                "sha256": _file_digest(path),
            }
        )
    return {"type": "directory", "files": files}


def _write_completion_marker(output: Path) -> None:
    snapshot = _completion_snapshot(output)
    if snapshot is None:
        raise OSError(f"output does not exist after fetch: {output}")
    marker = _completion_marker_path(output)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": _COMPLETION_MARKER_VERSION, "snapshot": snapshot}
    fd, temporary = tempfile.mkstemp(prefix=f".{marker.name}.", suffix=".tmp", dir=marker.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _already_fetched(dst: Path, kind: EntryKind, entry: DatasetEntry) -> bool:
    output = _output_path(entry, kind, dst)
    marker = _completion_marker_path(output)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if payload.get("version") != _COMPLETION_MARKER_VERSION:
        return False
    return payload.get("snapshot") == _completion_snapshot(output)


def _dispatch(
    entry: DatasetEntry,
    kind: EntryKind,
    dst: Path,
    *,
    hf_tok: str | None,
    gh_tok: str | None,
) -> Path:
    if kind == "hf":
        return _run_hf(entry, dst, token=hf_tok)
    if kind == "url_raw":
        return _run_raw(entry, dst, token=gh_tok)
    if kind == "url_github_tree":
        _run_tree(entry, dst, token=gh_tok)
        return dst
    if kind == "url_github_clone":
        _run_clone(entry, dst, token=gh_tok)
        return dst
    raise AssertionError(f"Unhandled entry kind: {kind!r}")


def _run_hf(entry: DatasetEntry, dst: Path, *, token: str | None) -> Path:
    from convmerge.fetch.hf import download_hf_dataset

    target = dst if dst.suffix else dst.with_suffix(".jsonl")
    return download_hf_dataset(
        entry.hf or "",
        target,
        config=entry.config,
        split=entry.split,
        token=token,
    )


def _run_raw(entry: DatasetEntry, dst: Path, *, token: str | None) -> Path:
    from convmerge.fetch.github import download_raw_file

    url = entry.url or ""
    suffix = _raw_suffix(url)
    target = dst if dst.suffix else dst.with_suffix(suffix)
    return download_raw_file(url, target, token=token)


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
