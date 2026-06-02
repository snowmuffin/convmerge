"""Deduplicate JSONL files by hashing a canonical projection of each record."""

from __future__ import annotations

import hashlib
import json
import os
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


class _MemorySeen:
    """In-memory ``set`` of seen hashes (default, fastest)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def add(self, h: str) -> bool:
        """Return True if ``h`` is new (and remember it), False if a duplicate."""
        if h in self._seen:
            return False
        self._seen.add(h)
        return True

    def close(self) -> None:  # pragma: no cover - trivial
        self._seen.clear()


class _SqliteSeen:
    """Disk-backed seen set for inputs with very high unique cardinality.

    Hashes live in a SQLite table instead of a Python ``set``, so peak RSS stays
    roughly constant regardless of how many unique rows are seen. Slower than
    the in-memory store but bounded in memory.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        import sqlite3
        import tempfile

        if path is None:
            fd, tmp = tempfile.mkstemp(prefix="convmerge-dedupe-", suffix=".sqlite")
            os.close(fd)
            self._path = tmp
            self._owns_file = True
        else:
            self._path = str(path)
            self._owns_file = False

        self._conn = sqlite3.connect(self._path)
        # Dedup state is disposable: durability is unnecessary, so trade it for
        # speed.
        self._conn.execute("PRAGMA journal_mode=OFF")
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.execute("CREATE TABLE IF NOT EXISTS seen (h TEXT PRIMARY KEY)")

    def add(self, h: str) -> bool:
        cur = self._conn.execute("INSERT OR IGNORE INTO seen(h) VALUES (?)", (h,))
        return cur.rowcount > 0

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()
        if self._owns_file:
            try:
                os.remove(self._path)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass


def _make_seen_store(kind: str, *, path: str | Path | None = None):
    if kind == "memory":
        return _MemorySeen()
    if kind == "sqlite":
        return _SqliteSeen(path)
    raise ValueError(f"Unknown seen_store {kind!r}. Choose 'memory' or 'sqlite'.")


def deduplicate_jsonl(
    src: str | Path,
    dst: str | Path,
    *,
    keys: Iterable[str] | None = None,
    algorithm: str | HashFn = "md5",
    progress: bool = False,
    seen_store: str = "memory",
    seen_db: str | Path | None = None,
) -> tuple[int, int]:
    """Stream ``src`` JSONL into ``dst``, dropping duplicate rows.

    A row's identity is the hash of ``json.dumps(projection, sort_keys=True)``.
    ``keys`` optionally restricts the projection to the given top-level fields
    (useful when metadata like ``id`` or timestamps differ but the content is
    identical). ``algorithm`` may be ``"md5"``, ``"sha256"``, or any callable
    ``bytes -> str``. Set ``progress=True`` to log periodic row counts to
    stderr (off by default; see :mod:`convmerge.progress`).

    ``seen_store`` selects how seen hashes are tracked:

    - ``"memory"`` (default): a Python ``set`` — fastest, but peak RAM grows
      with the number of *unique* rows.
    - ``"sqlite"``: a disk-backed SQLite table — bounded memory for inputs with
      tens of millions of unique rows, at some speed cost. ``seen_db`` chooses
      the database path; a temporary file is created and removed automatically
      when omitted.

    Returns ``(total_rows, kept_rows)``.
    """
    from convmerge.progress import ProgressReporter

    reporter = ProgressReporter(f"dedupe {Path(src).name}", enabled=progress)
    store = _make_seen_store(seen_store, path=seen_db)
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

    total = 0
    kept = 0
    try:
        with src_p.open(encoding="utf-8") as rf, dst_p.open("w", encoding="utf-8") as wf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                total += 1
                reporter.update()
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # Skip corrupt rows silently; caller can re-run normalize first.
                    continue
                projection = _project(data, key_set)
                normalized = json.dumps(projection, sort_keys=True, ensure_ascii=False)
                h = hasher(normalized.encode("utf-8"))
                if not store.add(h):
                    continue
                wf.write(line + "\n")
                kept += 1
    finally:
        store.close()
    reporter.done()
    return total, kept


def _project(data: Any, keys: set[str] | None) -> Any:
    if keys is None or not isinstance(data, dict):
        return data
    return {k: data[k] for k in keys if k in data}
