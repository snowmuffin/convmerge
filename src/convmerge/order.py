"""Deterministic curriculum ordering (``convmerge order``).

Rows are ranked by a key, ascending unless ``descending``:

- ``length``: characters of the conversation's text (a preference row: its
  prompt and chosen answer);
- ``turns``: user turns;
- ``meta:FIELD``: a number under the row's ``meta`` (for example a score
  ``convert --keep-fields`` carried there, or a difficulty label).

Ties are broken by a hash of the row and ``seed``, never by input order, so
the same rows in any order give the same file. With ``stages``, the ranked
rows are cut into that many stages of (nearly) equal size, easiest first,
and shuffled within each stage by the same seeded hash: the model sees the
stages in order without seeing near-identical rows back to back. Rows the
key cannot rank (no readable conversation, a missing or non-numeric
``meta`` value) go last, in hash order, and are counted.

Rows are written byte-for-byte; only their order changes. Memory holds one
small record per row (key, hash, file offset), not the rows. Most trainers
shuffle by default, so turn that off for the order to matter (see
``docs/format.md``).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from convmerge.io import refuse_overwrite

__all__ = ["OrderStats", "order_jsonl"]


@dataclass
class OrderStats:
    """Counts for one :func:`order_jsonl` call."""

    rows: int = 0
    unranked: int = 0
    """Rows the key could not rank; written last."""
    by: str = ""
    stages: list[dict[str, Any]] = field(default_factory=list)
    """Per stage: ``rows`` and the key range (``min``, ``max``) it covers."""

    def to_report(self) -> dict[str, Any]:
        return {"by": self.by, "rows": self.rows, "unranked": self.unranked,
                "stages": self.stages}  # fmt: skip


def order_jsonl(
    src: str | Path,
    dst: str | Path,
    *,
    by: str = "length",
    descending: bool = False,
    stages: int | None = None,
    seed: int = 0,
    encoding: str = "utf-8",
    stats: OrderStats | None = None,
) -> OrderStats:
    """Write ``src``'s rows to ``dst`` in curriculum order; see the module docstring."""
    import json

    from convmerge._output import atomic_text_writer
    from convmerge.adapter_resolve import resolve_adapter

    key = _key_function(by)
    if stages is not None and stages < 1:
        raise ValueError(f"stages must be at least 1, got {stages}")
    refuse_overwrite([src], [dst])
    st = stats if stats is not None else OrderStats()
    st.by = by
    adapter = resolve_adapter("auto", None, pairs=True)
    salt = str(seed).encode()

    ranked: list[tuple[float, bytes, int, float]] = []
    unranked: list[tuple[bytes, int]] = []
    with open(src, "rb") as f:
        offset = 0
        for raw in iter(f.readline, b""):
            start, offset = offset, offset + len(raw)
            line = raw.rstrip(b"\r\n")
            if start == 0 and line.startswith(b"\xef\xbb\xbf"):
                line = line[3:]
            if not line.strip():
                continue
            st.rows += 1
            tie = hashlib.sha256(salt + b"\0" + line).digest()
            try:
                value = json.loads(line.decode(encoding))
            except (UnicodeDecodeError, json.JSONDecodeError):
                value = None
            k = key(value, adapter) if isinstance(value, dict) else None
            if k is None:
                unranked.append((tie, start))
            else:
                ranked.append((-k if descending else k, tie, start, k))
    ranked.sort()
    unranked.sort()
    st.unranked = len(unranked)

    order = [start for _, _, start, _ in ranked]
    if stages is not None and ranked:
        order = []
        size = math.ceil(len(ranked) / stages)
        for i in range(0, len(ranked), size):
            part = ranked[i : i + size]
            keys = [k for _, _, _, k in part]
            st.stages.append({"rows": len(part), "min": min(keys), "max": max(keys)})
            order += [start for _, _, start, _ in sorted(part, key=lambda r: r[1])]
    order += [start for _, start in unranked]

    with open(src, "rb") as f, atomic_text_writer(dst) as out:
        for start in order:
            f.seek(start)
            line = f.readline().rstrip(b"\r\n")
            if start == 0 and line.startswith(b"\xef\xbb\xbf"):
                line = line[3:]
            out.write(line.decode(encoding, errors="replace") + "\n")
    return st


def _key_function(by: str) -> Any:
    if by == "length":
        return lambda value, adapter: _first(value, adapter, _chars)
    if by == "turns":
        return lambda value, adapter: _first(value, adapter, _user_turns)
    if by.startswith("meta:") and by[5:]:
        name = by[5:]

        def meta(value: dict[str, Any], adapter: Any) -> float | None:
            m = value.get("meta")
            v = m.get(name) if isinstance(m, dict) else None
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                return None  # missing, not a number, or NaN
            return float(v)

        return meta
    raise ValueError(f"order by: expected length, turns, or meta:FIELD, got {by!r}")


def _first(value: dict[str, Any], adapter: Any, measure: Any) -> float | None:
    examples = [ex for ex in adapter(value) if ex.messages]
    return float(measure(examples[0])) if examples else None


def _chars(ex: Any) -> int:
    return sum(len(m.text) for m in ex.messages)


def _user_turns(ex: Any) -> int:
    return sum(1 for m in ex.messages if m.role == "user")
