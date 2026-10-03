"""Near-duplicate removal with MinHash LSH (``convmerge dedupe --near``).

Each row's text (every turn of a conversation except the system prompt,
both sides of a preference pair; or the string values of ``keys``) is cut
into word ``shingle``-grams.
Rows whose estimated Jaccard similarity to an earlier kept row reaches
``threshold`` are dropped, so the first of a group of near-copies stays.
This catches the same source translated or reformatted slightly
differently, which exact ``dedupe`` cannot.

It is approximate (MinHash estimates similarity; LSH finds candidates) and
keeps its index in memory (a 60-bit digest per LSH band, about 0.7 KB a row
at the defaults), so it suits up to several million rows; for larger corpora
use a distributed tool such as datatrove. Needs
``pip install "convmerge[quality]"`` (datasketch).

The index uses datasketch's ``MinHashLSH`` band layout and band bytes, so a
row is a near-duplicate exactly when ``MinHashLSH.query`` would find an
earlier kept row (up to a 2**-60 chance of a digest collision per band).
"""

from __future__ import annotations

import hashlib
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
) -> tuple[int, int]:
    """Write the rows of ``src`` that are not near-copies of an earlier row to ``dst``.

    Returns ``(total, kept)`` like :func:`~convmerge.normalize.dedup.deduplicate_jsonl`.
    Rows are written byte-for-byte as read; invalid JSON lines are dropped.
    ``workers`` > 1 computes MinHashes in that many processes; the output is
    the same as with one.
    """
    refuse_overwrite([src], [dst, rejects])
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
    seen: set[int] = set()
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    rej = open(rejects, "w", encoding="utf-8") if rejects is not None else None
    invalid = ReadStats()
    try:
        with open(dst, "w", encoding="utf-8") as out:
            for raw, bands in _signed_rows(src, encoding, signer, workers, invalid):
                st.total += 1
                if not seen.isdisjoint(bands):
                    st.near_duplicates += 1
                    if rej is not None:
                        rej.write(raw + "\n")
                    continue
                seen.update(bands)
                st.kept += 1
                out.write(raw + "\n")
    finally:
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
