"""Order-preserving ``--workers`` processing of JSONL files.

The main process reads raw lines (with the decoding rules of
:func:`convmerge.io.iter_jsonl`) in chunks; worker processes parse and handle
each chunk with :func:`parse_chunk`, which applies the same line rules, and
the results come back in input order. A bounded window of chunks in flight
keeps memory flat on large inputs.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Protocol, TypeVar

from convmerge import io

CHUNK_LINES = 2_000

Chunk = tuple[list[tuple[int, str]], bool]
"""``([(line_number, raw), ...], check_bytes)``."""

T = TypeVar("T")


class LineCounts(Protocol):
    lines_read: int
    blank: int


class InvalidCounts(Protocol):
    invalid_json: int
    first_invalid_line: int | None


def raw_chunks(
    path: str | Path, encoding: str, counts: LineCounts, size: int | None = None
) -> Iterator[Chunk]:
    """Non-blank lines in chunks, counting lines and blanks in ``counts``.

    ``check_bytes`` is true once the file has shown an undecodable byte, as in
    :func:`convmerge.io.iter_jsonl`.
    """
    size = size or CHUNK_LINES
    before = io.bad_bytes_seen()
    chunk: list[tuple[int, str]] = []
    for number, raw in io.iter_raw_lines(path, encoding=encoding):
        counts.lines_read += 1
        if not raw:
            counts.blank += 1
            continue
        chunk.append((number, raw))
        if len(chunk) >= size:
            yield chunk, io.bad_bytes_seen() != before
            chunk = []
    if chunk:
        yield chunk, io.bad_bytes_seen() != before


def parse_chunk(
    chunk: Chunk, encoding: str, invalid: InvalidCounts
) -> Iterator[tuple[int, str, Any]]:
    """``(line_number, raw, value)`` for each usable line; the rest are counted."""
    lines, check_bytes = chunk
    for number, raw in lines:
        value, error = io._parse_line(raw, check_bytes=check_bytes, encoding=encoding)
        if error is not None:
            invalid.invalid_json += 1
            if invalid.first_invalid_line is None:
                invalid.first_invalid_line = number
            continue
        yield number, raw, value


def merge_invalid(into: InvalidCounts, part: InvalidCounts) -> None:
    """Add a later chunk's invalid-line counts to ``into``."""
    into.invalid_json += part.invalid_json
    if into.first_invalid_line is None:
        into.first_invalid_line = part.first_invalid_line


def ordered_map(
    chunks: Iterator[Chunk],
    work: Callable[[Chunk], T],
    *,
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
) -> Iterator[T]:
    """``work(chunk)`` for every chunk in ``workers`` processes, yielded in order."""
    window = workers * 4
    with ProcessPoolExecutor(workers, initializer=initializer, initargs=initargs) as pool:
        pending: deque = deque()
        for chunk in chunks:
            pending.append(pool.submit(work, chunk))
            if len(pending) >= window:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
