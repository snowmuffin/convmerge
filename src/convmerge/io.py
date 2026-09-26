"""Shared line-by-line JSONL reader.

Every command that streams JSONL (convert, dedupe, mix, turns, load_jsonl)
reads through :func:`iter_jsonl`, so blank lines, a UTF-8 BOM, and
unparseable lines are handled — and counted — the same way everywhere.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

OnError = Literal["skip", "raise"]


class JsonlDecodeError(ValueError):
    """A JSONL line failed to parse (raised when ``on_error="raise"``)."""

    def __init__(self, path: str | Path, line_number: int, error: json.JSONDecodeError, raw: str):
        super().__init__(f"{path} line {line_number}: {error.msg} :: {raw[:80]!r}")
        self.path = str(path)
        self.line_number = line_number
        self.raw = raw


@dataclass
class ReadStats:
    """Counters filled by :func:`iter_jsonl` when ``stats`` is passed."""

    lines_read: int = 0
    blank: int = 0
    invalid_json: int = 0
    first_invalid_line: int | None = None


@dataclass(frozen=True)
class JsonlLine:
    """One parsed, non-blank line."""

    number: int
    """1-based physical line number."""
    raw: str
    """The line with surrounding whitespace (and a leading BOM) removed."""
    value: Any
    """The parsed JSON value (any type; callers decide what they accept)."""


def iter_jsonl(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    on_error: OnError = "skip",
    stats: ReadStats | None = None,
    on_invalid: Callable[[JsonlDecodeError], None] | None = None,
) -> Iterator[JsonlLine]:
    """Yield each non-blank line of a JSONL file with its parsed value.

    ``on_error="skip"`` (default) drops unparseable lines and counts them in
    ``stats``, calling ``on_invalid`` (if given) with the error for each;
    ``"raise"`` raises :class:`JsonlDecodeError` at the first one.
    A UTF-8 byte-order mark on the first line is ignored.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
    st = stats if stats is not None else ReadStats()
    with Path(path).open(encoding=encoding) as f:
        for number, line in enumerate(f, 1):
            st.lines_read += 1
            raw = line.strip()
            if number == 1:
                raw = raw.removeprefix("\ufeff").strip()
            if not raw:
                st.blank += 1
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as e:
                err = JsonlDecodeError(path, number, e, raw)
                if on_error == "raise":
                    raise err from None
                if on_invalid is not None:
                    on_invalid(err)
                st.invalid_json += 1
                if st.first_invalid_line is None:
                    st.first_invalid_line = number
                continue
            yield JsonlLine(number=number, raw=raw, value=value)
