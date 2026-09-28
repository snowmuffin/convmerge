"""Near-duplicate removal with MinHash LSH (``convmerge dedupe --near``).

Each row's text (every turn of a conversation, both sides of a preference
pair; or the string values of ``keys``) is cut into word ``shingle``-grams.
Rows whose estimated Jaccard similarity to an earlier kept row reaches
``threshold`` are dropped, so the first of a group of near-copies stays.
This catches the same source translated or reformatted slightly
differently, which exact ``dedupe`` cannot.

It is approximate (MinHash estimates similarity; LSH finds candidates) and
keeps its index in memory, so it suits up to a few million rows; for larger
corpora use a distributed tool such as datatrove. Needs
``pip install "convmerge[quality]"`` (datasketch).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from convmerge._text import ngrams, words
from convmerge.io import ReadStats, iter_jsonl


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
) -> tuple[int, int]:
    """Write the rows of ``src`` that are not near-copies of an earlier row to ``dst``.

    Returns ``(total, kept)`` like :func:`~convmerge.normalize.dedup.deduplicate_jsonl`.
    Rows are written byte-for-byte as read; invalid JSON lines are dropped.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold: expected a fraction between 0 and 1")
    if num_perm < 16 or shingle < 1:
        raise ValueError("num_perm must be at least 16 and shingle positive")
    ds = _require_datasketch()
    from convmerge.adapter_resolve import resolve_adapter

    st = stats if stats is not None else NearDedupeStats()
    st.threshold = threshold
    key_list = list(keys) if keys else None
    adapter = resolve_adapter("auto", None, pairs=True)
    try:
        lsh = ds.MinHashLSH(threshold=threshold, num_perm=num_perm)
    except ValueError as e:
        raise ValueError(f"threshold {threshold} with num_perm {num_perm}: {e}; "
                         "lower the threshold or raise num_perm") from None  # fmt: skip
    read = ReadStats()
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    rej = open(rejects, "w", encoding=encoding) if rejects is not None else None
    try:
        with open(dst, "w", encoding=encoding) as out:
            for line in iter_jsonl(src, encoding=encoding, stats=read):
                st.total += 1
                text = _row_text(line.value, adapter, key_list)
                mh = ds.MinHash(num_perm=num_perm)
                mh.update_batch([s.encode("utf-8") for s in _shingles(text, shingle)])
                if lsh.query(mh):
                    st.near_duplicates += 1
                    if rej is not None:
                        rej.write(line.raw + "\n")
                    continue
                lsh.insert(line.number, mh, check_duplication=False)
                st.kept += 1
                out.write(line.raw + "\n")
    finally:
        if rej is not None:
            rej.close()
        st.invalid_json = read.invalid_json
        st.first_invalid_line = read.first_invalid_line
        st.total += read.invalid_json
    return st.total, st.kept


def _row_text(value: Any, adapter: Any, keys: list[str] | None) -> str:
    if isinstance(value, dict) and keys is None:
        parts: list[str] = []
        for ex in adapter(value):
            for m in (*ex.messages, *(ex.rejected or ())):
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
