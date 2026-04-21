"""Resume-safety primitives for append-only JSONL pipelines.

Long-running jobs that append to a JSONL file (e.g. batch inference, chunked
evaluation) can crash mid-write, leaving a truncated last line. The helpers
here let resumer code:

1. Count how many rows are currently committed (``count_lines``).
2. Trim off a corrupt trailing line so the next run re-runs that row from
   scratch (``trim_corrupt_tail``).
3. Keep several parallel output files in lockstep by truncating all of them
   to the same prefix length (``truncate_to_n_lines``).

None of these helpers depend on a particular pipeline or remote service.
"""

from __future__ import annotations

import json
from pathlib import Path


def count_lines(path: str | Path) -> int:
    """Number of ``\\n``-terminated lines in a file. Missing file -> 0."""
    p = Path(path)
    if not p.is_file():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def truncate_to_n_lines(path: str | Path, n: int) -> None:
    """Keep only the first ``n`` lines of ``path``. ``n == 0`` clears the file.

    Missing file is a no-op. The file is rewritten in full (safe for
    moderate sizes; not optimised for multi-GB tails).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    p = Path(path)
    if not p.is_file():
        return
    if n == 0:
        p.write_text("", encoding="utf-8")
        return
    with p.open(encoding="utf-8") as f:
        lines = f.readlines()
    with p.open("w", encoding="utf-8") as f:
        f.writelines(lines[:n])


def trim_corrupt_tail(
    path: str | Path,
    *,
    companions: tuple[str | Path, ...] = (),
) -> int:
    """Drop a truncated JSON line at the end of ``path``.

    Walks the file top-to-bottom, parsing each non-empty line. The first
    line that fails to parse (and every line after it, if any) is removed.
    This is the canonical "resume from the last good row" recipe for jobs
    that append one JSON object per row.

    If ``companions`` are supplied, each of them is truncated to the same
    line count as ``path`` after the tail is trimmed. This keeps several
    parallel output files aligned (e.g. a bundle file plus one file per
    downstream split).

    Returns the number of valid lines retained in ``path``.
    """
    p = Path(path)
    if not p.is_file():
        return 0
    with p.open(encoding="utf-8") as f:
        lines = f.readlines()

    n_raw = len(lines)
    n_ok = 0
    for line in lines:
        s = line.rstrip("\r\n")
        if not s.strip():
            # Preserve blank-line accounting as in the raw file.
            n_ok += 1
            continue
        try:
            json.loads(s)
        except json.JSONDecodeError:
            break
        n_ok += 1

    if n_ok < n_raw:
        truncate_to_n_lines(p, n_ok)

    for companion in companions:
        cp = Path(companion)
        if cp.is_file() and count_lines(cp) > n_ok:
            truncate_to_n_lines(cp, n_ok)

    return n_ok
