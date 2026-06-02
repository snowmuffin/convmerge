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


def profile_schema(
    records: Iterable[dict[str, Any]] | str | Path,
    *,
    max_rows: int | None = None,
    max_examples: int = 3,
) -> dict[str, Any]:
    """Infer a structural profile of JSON/JSONL records for mapping design.

    Unlike :func:`key_frequency`, this preserves nesting: list-of-object fields
    (e.g. ``messages``) report their inner fields under ``items``, and object
    fields under ``fields``, so ``messages[].role`` is distinguishable from a
    top-level ``role``. For each field it reports value ``types`` (counts),
    ``present`` (how many parent records contained it), ``presence`` (that as a
    fraction), and a few scalar ``examples``.

    Accepts an iterable of dicts or a path to a ``.json`` / ``.jsonl`` file.
    Pass ``max_rows`` to sample only the first N records (recommended for large
    files, since profiling materializes the sampled records in memory).
    """
    recs = [r for r in _as_record_iter(records, max_rows=max_rows) if isinstance(r, dict)]
    total = len(recs)

    base: set[str] | None = None
    uniform = total > 0
    for r in recs:
        keys = set(r.keys())
        if base is None:
            base = keys
        elif keys != base:
            uniform = False
            break

    return {
        "records": total,
        "uniform_top_level": uniform,
        "fields": _profile_records(recs, max_examples=max_examples),
    }


def _profile_records(records: list[dict[str, Any]], *, max_examples: int) -> dict[str, Any]:
    total = len(records)
    values_by_key: dict[str, list[Any]] = {}
    present_by_key: Counter[str] = Counter()
    for rec in records:
        for k, v in rec.items():
            values_by_key.setdefault(k, []).append(v)
            present_by_key[k] += 1
    return {
        k: _profile_field(
            vals,
            present=present_by_key[k],
            total=total,
            max_examples=max_examples,
        )
        for k, vals in values_by_key.items()
    }


def _profile_field(
    values: list[Any],
    *,
    present: int,
    total: int,
    max_examples: int,
) -> dict[str, Any]:
    types: Counter[str] = Counter()
    examples: list[Any] = []
    child_dicts: list[dict[str, Any]] = []
    list_item_dicts: list[dict[str, Any]] = []

    for v in values:
        types[_type_name(v)] += 1
        if isinstance(v, dict):
            child_dicts.append(v)
        elif isinstance(v, list):
            list_item_dicts.extend(item for item in v if isinstance(item, dict))
        elif v is not None and len(examples) < max_examples:
            ex = _example(v)
            if ex not in examples:
                examples.append(ex)

    profile: dict[str, Any] = {
        "types": dict(types),
        "present": present,
        "presence": round(present / total, 3) if total else 0.0,
    }
    if examples:
        profile["examples"] = examples
    if child_dicts:
        profile["fields"] = _profile_records(child_dicts, max_examples=max_examples)
    if list_item_dicts:
        profile["items"] = _profile_records(list_item_dicts, max_examples=max_examples)
    return profile


def _type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):  # bool before int (bool is an int subclass)
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _example(v: Any, *, limit: int = 80) -> Any:
    if isinstance(v, str) and len(v) > limit:
        return v[: limit - 3] + "..."
    return v


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
