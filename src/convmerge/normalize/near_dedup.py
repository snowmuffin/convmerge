"""Near-duplicate removal with MinHash LSH (``convmerge dedupe --near``).

Each row's text (every turn of a conversation except the system prompt,
both sides of a preference pair; or the string values of ``keys``) is cut
into word ``shingle``-grams.
Rows whose estimated Jaccard similarity to an earlier kept row reaches
``threshold`` are dropped, so the first of a group of near-copies stays.
This catches the same source translated or reformatted slightly
differently, which exact ``dedupe`` cannot.

It is approximate (MinHash estimates similarity; LSH finds candidates). By
default the index lives in memory (a 60-bit digest per LSH band, about 0.7 KB
a kept row at the defaults), which suits up to several million rows;
``seen_store="sqlite"`` keeps it in a disk-backed table instead, with memory
bounded by SQLite's page cache and the same output. Needs
``pip install "convmerge[quality]"`` (datasketch).

The index uses datasketch's ``MinHashLSH`` band layout and band bytes, so a
row is a near-duplicate exactly when ``MinHashLSH.query`` would find an
earlier kept row (up to a 2**-60 chance of a digest collision per band).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from convmerge import _parallel
from convmerge._text import ngrams, words
from convmerge.io import ReadStats, iter_jsonl, refuse_overwrite


@dataclass
class NearDedupeStats:
    """Counters filled by :func:`deduplicate_near_jsonl`.

    ``total`` counts non-blank lines; each is ``kept``, a ``near_duplicates``
    hit, or ``invalid_json`` (unparseable, dropped).
    """

    total: int = 0
    kept: int = 0
    near_duplicates: int = 0
    invalid_json: int = 0
    first_invalid_line: int | None = None
    threshold: float = 0.8


# Page cache of the sqlite band store: the memory it uses however many rows
# are kept. Pages beyond it are written to the database file and read back.
_SQLITE_CACHE_KIB = 64 * 1024


class _MemoryBands:
    """Every kept row's band keys in a ``set`` (fastest; memory grows with kept rows)."""

    def __init__(self) -> None:
        self._seen: set[int] = set()

    def add_if_new(self, bands: list[int]) -> bool:
        """Remember ``bands`` and return True unless one of them was seen before."""
        if not self._seen.isdisjoint(bands):
            return False
        self._seen.update(bands)
        return True

    def close(self) -> None:
        self._seen.clear()


class _SqliteBands:
    """The same lookups against a SQLite table of every kept row's band keys.

    Every key ever added stays queryable however much of the table has been
    written out of the page cache, so the kept/dropped decisions are those of
    :class:`_MemoryBands`. Without ``path`` the database is a temporary file
    removed by :meth:`close`; a given ``path`` is kept, and keys already in it
    count as seen (deduplicating against an earlier run).
    """

    def __init__(self, path: str | Path | None = None) -> None:
        import sqlite3

        if path is None:
            fd, tmp = tempfile.mkstemp(prefix="convmerge-near-", suffix=".sqlite")
            os.close(fd)
            self._path, self._owns_file = tmp, True
        else:
            self._path, self._owns_file = str(path), False
        try:
            self._conn = sqlite3.connect(self._path)
            # The index is disposable: trade durability for speed.
            self._conn.execute("PRAGMA journal_mode=OFF")
            self._conn.execute("PRAGMA synchronous=OFF")
            self._conn.execute(f"PRAGMA cache_size=-{_SQLITE_CACHE_KIB}")
            self._conn.execute("CREATE TABLE IF NOT EXISTS near_bands (k INTEGER PRIMARY KEY)")
        except BaseException:
            self._remove()
            raise
        self._query = ""

    def add_if_new(self, bands: list[int]) -> bool:
        if len(bands) != self._query.count("?"):
            marks = ",".join("?" * len(bands))
            self._query = f"SELECT 1 FROM near_bands WHERE k IN ({marks}) LIMIT 1"
        if self._conn.execute(self._query, bands).fetchone() is not None:
            return False
        self._conn.executemany("INSERT OR IGNORE INTO near_bands(k) VALUES (?)",
                               [(k,) for k in bands])  # fmt: skip
        return True

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()
            self._remove()

    def _remove(self) -> None:
        if self._owns_file:
            try:
                os.remove(self._path)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass


def _band_store(kind: str, path: str | Path | None) -> _MemoryBands | _SqliteBands:
    if kind == "memory":
        if path is not None:
            raise ValueError("seen_db needs seen_store='sqlite'")
        return _MemoryBands()
    if kind == "sqlite":
        return _SqliteBands(path)
    raise ValueError(f"Unknown seen_store {kind!r}. Choose 'memory' or 'sqlite'.")


def _require_datasketch() -> Any:
    try:
        import datasketch
    except ImportError as e:
        raise ImportError(
            "near-duplicate removal needs datasketch: pip install 'convmerge[quality]' (or [all])"
        ) from e
    return datasketch


def deduplicate_near_jsonl(
    src: str | Path,
    dst: str | Path,
    *,
    threshold: float = 0.8,
    num_perm: int = 128,
    shingle: int = 5,
    keys: Iterable[str] | None = None,
    rejects: str | Path | None = None,
    encoding: str = "utf-8",
    stats: NearDedupeStats | None = None,
    workers: int = 1,
    seen_store: str = "memory",
    seen_db: str | Path | None = None,
) -> tuple[int, int]:
    """Write the rows of ``src`` that are not near-copies of an earlier row to ``dst``.

    Returns ``(total, kept)`` like :func:`~convmerge.normalize.dedup.deduplicate_jsonl`.
    Rows are written byte-for-byte as read; invalid JSON lines are dropped.
    ``workers`` > 1 computes MinHashes in that many processes; the output is
    the same as with one.

    ``seen_store`` selects where the kept rows' band keys live:

    - ``"memory"`` (default): a Python ``set``, fastest; memory grows by about
      0.7 KB per kept row at the defaults.
    - ``"sqlite"``: a disk-backed SQLite table; memory stays near the page
      cache (64 MiB) and the disk file grows instead (about 140 bytes per kept
      row). The output is the same as with ``"memory"``. ``seen_db`` chooses
      the database path; without it a temporary file is created in the
      system temporary directory and removed at the end, also on failure.
      Keys already in a given ``seen_db`` count as seen.
    """
    refuse_overwrite([src], [dst, rejects, seen_db])
    if not 0 < threshold < 1:
        raise ValueError("threshold: expected a fraction between 0 and 1")
    if num_perm < 16 or shingle < 1:
        raise ValueError("num_perm must be at least 16 and shingle positive")
    if workers < 1:
        raise ValueError("workers: expected a positive integer")
    ds = _require_datasketch()
    try:
        lsh = ds.MinHashLSH(threshold=threshold, num_perm=num_perm)
    except ValueError as e:
        raise ValueError(f"threshold {threshold} with num_perm {num_perm}: {e}; "
                         "lower the threshold or raise num_perm") from None  # fmt: skip
    st = stats if stats is not None else NearDedupeStats()
    st.threshold = threshold
    signer = _Signer(num_perm, lsh.b, lsh.r, shingle, list(keys) if keys else None)
    seen = _band_store(seen_store, seen_db)
    rej = None
    invalid = ReadStats()
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        rej = open(rejects, "w", encoding="utf-8") if rejects is not None else None
        with open(dst, "w", encoding="utf-8") as out:
            for raw, bands in _signed_rows(src, encoding, signer, workers, invalid):
                st.total += 1
                if not seen.add_if_new(bands):
                    st.near_duplicates += 1
                    if rej is not None:
                        rej.write(raw + "\n")
                    continue
                st.kept += 1
                out.write(raw + "\n")
    finally:
        seen.close()
        if rej is not None:
            rej.close()
        st.invalid_json = invalid.invalid_json
        st.first_invalid_line = invalid.first_invalid_line
        st.total += invalid.invalid_json
    return st.total, st.kept


class _Signer:
    """A row's LSH band keys: 60-bit digests of ``MinHashLSH``'s band bytes."""

    def __init__(self, num_perm: int, b: int, r: int, shingle: int, keys: list[str] | None) -> None:
        self.num_perm, self.b, self.r, self.shingle, self.keys = num_perm, b, r, shingle, keys
        self._ds: Any = None
        self._adapter: Any = None

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_ds": None, "_adapter": None}

    def bands(self, value: Any) -> list[int]:
        if self._ds is None:
            from convmerge.adapter_resolve import resolve_adapter

            self._ds = _require_datasketch()
            self._adapter = resolve_adapter("auto", None, pairs=True)
        text = _row_text(value, self._adapter, self.keys)
        mh = self._ds.MinHash(num_perm=self.num_perm)
        mh.update_batch([s.encode("utf-8") for s in _shingles(text, self.shingle)])
        hashes, r = mh.hashvalues, self.r
        # MinHashLSH keys band i by bytes(hashvalues[i*r:(i+1)*r].byteswap().data).
        return [
            int.from_bytes(
                hashlib.blake2b(
                    bytes(hashes[i * r : (i + 1) * r].byteswap().data),
                    digest_size=8, salt=i.to_bytes(8, "little"),
                ).digest(), "little",
            ) >> 4
            for i in range(self.b)
        ]  # fmt: skip


def _signed_rows(
    src: str | Path, encoding: str, signer: _Signer, workers: int, invalid: ReadStats
) -> Iterator[tuple[str, list[int]]]:
    """``(raw, band keys)`` for each parseable row, in order."""
    if workers == 1:
        for line in iter_jsonl(src, encoding=encoding, stats=invalid):
            yield line.raw, signer.bands(line.value)
        return
    chunks = _parallel.raw_chunks(src, encoding, ReadStats())
    parts = _parallel.ordered_map(
        chunks, _sign_chunk, workers=workers, initializer=_worker_init,
        initargs=(signer, encoding),
    )  # fmt: skip
    for rows, part in parts:
        _parallel.merge_invalid(invalid, part)
        yield from rows


_WORKER: dict[str, Any] = {}


def _worker_init(signer: _Signer, encoding: str) -> None:
    _WORKER.update(signer=signer, encoding=encoding)


def _sign_chunk(chunk: _parallel.Chunk) -> tuple[list[tuple[str, list[int]]], ReadStats]:
    part = ReadStats()
    signer: _Signer = _WORKER["signer"]
    rows = [
        (raw, signer.bands(value))
        for _number, raw, value in _parallel.parse_chunk(chunk, _WORKER["encoding"], part)
    ]
    return rows, part


def _row_text(value: Any, adapter: Any, keys: list[str] | None) -> str:
    if isinstance(value, dict) and keys is None:
        parts: list[str] = []
        for ex in adapter(value):
            # System prompts are often shared templates (tool definitions,
            # personas) that would make unrelated rows look alike.
            for m in (*ex.messages, *(ex.rejected or ())):
                if m.role != "system":
                    parts.append(m.text)
        if any(parts):
            return "\n".join(parts)
    if isinstance(value, dict) and keys is not None:
        value = [value.get(k) for k in keys]
    return "\n".join(_strings([value]))


def _strings(values: Iterable[Any]) -> Iterator[str]:
    for v in values:
        if isinstance(v, str):
            yield v
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield str(v)
        elif isinstance(v, (list, tuple)):
            yield from _strings(v)
        elif isinstance(v, dict):
            yield from _strings(v.values())


def _shingles(text: str, n: int) -> set[str]:
    tokens = words(text)
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(g) for g in ngrams(tokens, n)}
