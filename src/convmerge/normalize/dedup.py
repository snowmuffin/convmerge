"""Deduplicate JSONL files by hashing a canonical projection of each record."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

HashFn = Callable[[bytes], str]


def _md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_BUILTIN_HASHES: dict[str, HashFn] = {
    "md5": _md5_hex,
    "sha256": _sha256_hex,
}


def deduplicate_jsonl(
    src: str | Path,
    dst: str | Path,
    *,
    keys: Iterable[str] | None = None,
    algorithm: str | HashFn = "md5",
) -> tuple[int, int]:
    """Stream ``src`` JSONL into ``dst``, dropping duplicate rows.

    A row's identity is the hash of ``json.dumps(projection, sort_keys=True)``.
    ``keys`` optionally restricts the projection to the given top-level fields
    (useful when metadata like ``id`` or timestamps differ but the content is
    identical). ``algorithm`` may be ``"md5"``, ``"sha256"``, or any callable
    ``bytes -> str``.

    Returns ``(total_rows, kept_rows)``.
    """
    if callable(algorithm):
        hasher = algorithm
    else:
        try:
            hasher = _BUILTIN_HASHES[algorithm]
        except KeyError as e:
            raise ValueError(
                f"Unknown hash algorithm {algorithm!r}. "
                f"Choose one of: {sorted(_BUILTIN_HASHES)} or pass a callable."
            ) from e

    key_set = set(keys) if keys is not None else None

    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    total = 0
    kept = 0
    with src_p.open(encoding="utf-8") as rf, dst_p.open("w", encoding="utf-8") as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Skip corrupt rows silently; caller can re-run normalize first.
                continue
            projection = _project(data, key_set)
            normalized = json.dumps(projection, sort_keys=True, ensure_ascii=False)
            h = hasher(normalized.encode("utf-8"))
            if h in seen:
                continue
            seen.add(h)
            wf.write(line + "\n")
            kept += 1
    return total, kept


def _project(data: Any, keys: set[str] | None) -> Any:
    if keys is None or not isinstance(data, dict):
        return data
    return {k: data[k] for k in keys if k in data}
