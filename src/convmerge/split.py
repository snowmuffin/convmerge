"""Train / validation split of a JSONL file (``convmerge split``).

Which side a row lands on depends only on its **content** and the seed —
never on its position — so a split is reproducible, stable when rows are
added or reordered, and exact duplicates always land on the same side (no
train→validation leakage through copies). ``keys`` hashes only those
top-level fields, so for example every answer to one ``prompt`` stays
together.

- ``val=0.05``: each row goes to validation when its hash falls in the
  lowest 5% of the hash range. One streaming pass; the validation size is
  about 5% (not exact).
- ``val_rows=1000``: the 1,000 rows with the lowest hashes go to
  validation. Exact size, two passes, memory for ``val_rows`` hashes.

Rows are written as read (surrounding whitespace trimmed). Blank lines are
skipped; lines that are not valid JSON are dropped and counted.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from convmerge.io import ReadStats, iter_jsonl

_HASH_SPACE = 1 << 64


@dataclass
class SplitStats:
    """Counters filled by :func:`split_jsonl` when ``stats`` is passed."""

    total: int = 0
    train: int = 0
    val: int = 0
    invalid_json: int = 0
    first_invalid_line: int | None = None


def split_jsonl(
    src: str | Path,
    train_out: str | Path,
    val_out: str | Path,
    *,
    val: float | None = None,
    val_rows: int | None = None,
    seed: int = 42,
    keys: Iterable[str] | None = None,
    encoding: str = "utf-8",
    stats: SplitStats | None = None,
) -> tuple[int, int]:
    """Split ``src`` into ``train_out`` and ``val_out``; returns (train, val) row counts.

    Give exactly one of ``val`` (a fraction in ``(0, 1)``) or ``val_rows``
    (an exact count). See the module docstring for how rows are assigned.
    """
    if (val is None) == (val_rows is None):
        raise ValueError("give exactly one of val (a fraction) or val_rows (a count)")
    if val is not None and not 0.0 < val < 1.0:
        raise ValueError(f"val must be between 0 and 1 (exclusive), got {val}")
    if val_rows is not None and val_rows < 0:
        raise ValueError(f"val_rows must be >= 0, got {val_rows}")
    key_list = list(keys) if keys else None
    salt = f"convmerge-split:{seed}:".encode()
    st = stats if stats is not None else SplitStats()

    def row_hash(value: Any) -> int:
        return _hash(salt, value, key_list)

    if val is not None:
        threshold = int(val * _HASH_SPACE)

        def to_val(index: int, h: int) -> bool:
            return h < threshold

    else:
        chosen = _lowest(src, encoding, row_hash, val_rows or 0)

        def to_val(index: int, h: int) -> bool:
            return index in chosen

    read = ReadStats()
    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(val_out).parent.mkdir(parents=True, exist_ok=True)
    with (
        open(train_out, "w", encoding=encoding) as ftrain,
        open(val_out, "w", encoding=encoding) as fval,
    ):
        for index, line in enumerate(iter_jsonl(src, encoding=encoding, stats=read)):
            if to_val(index, row_hash(line.value)):
                fval.write(line.raw + "\n")
                st.val += 1
            else:
                ftrain.write(line.raw + "\n")
                st.train += 1
    st.invalid_json = read.invalid_json
    st.first_invalid_line = read.first_invalid_line
    st.total = st.train + st.val + st.invalid_json
    return st.train, st.val


def default_val_path(output: str | Path) -> Path:
    """Where the validation rows go by default: ``train.jsonl`` -> ``train.val.jsonl``."""
    out = Path(output)
    return out.with_name(f"{out.stem}.val{out.suffix or '.jsonl'}")


def _hash(salt: bytes, value: Any, keys: list[str] | None) -> int:
    if keys is not None:
        value = {k: value.get(k) for k in keys} if isinstance(value, dict) else value
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(salt + canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _lowest(src: str | Path, encoding: str, row_hash: Any, k: int) -> set[int]:
    """Indices of the ``k`` rows with the lowest hashes (ties broken by position)."""
    if k == 0:
        return set()
    heap: list[tuple[int, int]] = []  # max-heap of (-hash, -index)
    for index, line in enumerate(iter_jsonl(src, encoding=encoding)):
        item = (-row_hash(line.value), -index)
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return {-i for _, i in heap}
