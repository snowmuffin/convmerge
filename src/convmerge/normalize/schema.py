"""Uniform-schema detection and key-frequency counting for JSON/JSONL files."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from convmerge.normalize.jsonl import iter_json_records


def is_uniform_schema(
    records: Iterable[dict[str, Any]] | str | Path,
    *,
    max_rows: int | None = None,
) -> bool:
    """Check whether every record in ``records`` has the same top-level keys.

    Accepts either an iterable of dicts or a path to a ``.json`` / ``.jsonl``
    file. Returns ``True`` iff every record is a dict and shares the exact key
    set of the first dict encountered.
    """
    iterator = _as_record_iter(records, max_rows=max_rows)
    base: set[str] | None = None
    for item in iterator:
        if not isinstance(item, dict):
            return False
        keys = set(item.keys())
        if base is None:
            base = keys
            continue
        if keys != base:
            return False
    return base is not None


def key_frequency(
    records: Iterable[dict[str, Any]] | str | Path,
    *,
    recursive: bool = False,
    max_rows: int | None = None,
) -> dict[str, int]:
    """Count how often each key name appears across records.

    With ``recursive=True``, keys nested inside dict/list values are counted too
    (useful for discovering fields like ``messages[*].role``).
    """
    counts: Counter[str] = Counter()
    for rec in _as_record_iter(records, max_rows=max_rows):
        if not isinstance(rec, dict):
            continue
        _tally_keys(rec, counts, recursive=recursive)
    return dict(counts)


def _tally_keys(obj: Any, counts: Counter[str], *, recursive: bool) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            counts[k] += 1
            if recursive:
                _tally_keys(v, counts, recursive=recursive)
    elif recursive and isinstance(obj, list):
        for item in obj:
            _tally_keys(item, counts, recursive=recursive)


def _as_record_iter(
    records: Iterable[dict[str, Any]] | str | Path,
    *,
    max_rows: int | None,
) -> Iterable[dict[str, Any]]:
    if isinstance(records, (str, Path)):
        return iter_json_records(records, max_rows=max_rows)
    if max_rows is None:
        return records
    return _limit(records, max_rows)


def _limit(it: Iterable[dict[str, Any]], n: int) -> Iterable[dict[str, Any]]:
    for i, item in enumerate(it):
        if i >= n:
            return
        yield item
