"""Merge multiple JSONL files (or whole trees) into a single JSONL file."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def merge_jsonl(
    sources: Iterable[str | Path],
    dst: str | Path,
    *,
    skip_blank: bool = True,
    validate: bool = False,
) -> int:
    """Concatenate ``sources`` into a single JSONL file at ``dst``.

    Missing source files are silently skipped so callers can pass an
    optimistic list. With ``validate=True`` every non-blank line is parsed
    first; unparsable lines are dropped instead of being copied verbatim
    (useful when upstream files may contain binary garbage).

    Returns the number of lines written.
    """
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with dst_p.open("w", encoding="utf-8") as wf:
        for s in sources:
            src_p = Path(s)
            if not src_p.is_file():
                continue
            with src_p.open(encoding="utf-8") as rf:
                for raw in rf:
                    line = raw.strip()
                    if not line:
                        if skip_blank:
                            continue
                        wf.write("\n")
                        total += 1
                        continue
                    if validate:
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            continue
                    wf.write(line + "\n")
                    total += 1
    return total


def collect_jsonl_tree(
    source_dirs: Iterable[str | Path],
    dst: str | Path,
    *,
    extensions: Iterable[str] = (".jsonl",),
    validate: bool = True,
) -> tuple[int, int]:
    """Collect every ``*.jsonl`` file under ``source_dirs`` into one JSONL.

    Each directory is walked recursively; files whose suffix matches
    ``extensions`` (case-insensitive) are concatenated into ``dst``. Missing
    directories are skipped (with a stderr-style warning kept out of stdout)
    so callers can hand in an optimistic list.

    Returns ``(written, skipped_broken)`` — how many lines landed in ``dst``
    and how many lines were dropped because they did not parse as JSON
    (only counted when ``validate=True``).
    """
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.lower() for e in extensions)
    files: list[Path] = []
    for d in source_dirs:
        d_p = Path(d)
        if not d_p.is_dir():
            continue
        for p in d_p.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)

    written = 0
    skipped = 0
    with dst_p.open("w", encoding="utf-8") as wf:
        for path in files:
            with path.open(encoding="utf-8") as rf:
                for raw in rf:
                    line = raw.strip()
                    if not line:
                        continue
                    if validate:
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            skipped += 1
                            continue
                    wf.write(line + "\n")
                    written += 1
    return written, skipped
