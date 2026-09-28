"""Shared line-by-line JSONL reader.

Every command that streams JSONL (convert, dedupe, mix, turns, load_jsonl)
reads through :func:`iter_jsonl`, so blank lines, a UTF-8 BOM, and
unparseable lines are handled — and counted — the same way everywhere.
A line with bytes that are not valid in the file's encoding, an unpaired
UTF-16 surrogate escape (``"\\ud800"``), or nesting too deep to parse is
unparseable too: it is skipped like broken JSON instead of stopping the run.
"""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

OnError = Literal["skip", "raise"]


class JsonlDecodeError(ValueError):
    """A JSONL line failed to parse (raised when ``on_error="raise"``)."""

    def __init__(
        self, path: str | Path, line_number: int, error: json.JSONDecodeError | str, raw: str
    ):
        msg = error.msg if isinstance(error, json.JSONDecodeError) else error
        super().__init__(f"{path} line {line_number}: {msg} :: {raw[:80]!r}")
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


# ``\\uD800``..``\\uDFFF`` escape; lines containing one are checked for an
# unpaired surrogate after parsing (pairs such as emoji are fine).
_SURROGATE_ESCAPE = re.compile(r"(?i)\\ud[89a-f]")
# Undecodable bytes, read back as U+DC80..U+DCFF by the error handler below.
_ESCAPED_BYTE = re.compile("[\udc80-\udcff]")
_bad_bytes_seen = 0


def _mark_bad_bytes(err: UnicodeError) -> tuple[str | bytes, int]:
    # ``surrogateescape`` that also notes it fired, so only files that have
    # undecodable bytes pay for scanning each line for them.
    global _bad_bytes_seen
    _bad_bytes_seen += 1
    return codecs.lookup_error("surrogateescape")(err)


codecs.register_error("convmerge.surrogateescape", _mark_bad_bytes)


def _encoding_problem(raw: str, value: Any, *, check_bytes: bool, encoding: str) -> str | None:
    """Why the line cannot be written back out as UTF-8, or ``None``."""
    if check_bytes and _ESCAPED_BYTE.search(raw):
        return f"invalid {encoding} bytes"
    if _SURROGATE_ESCAPE.search(raw):
        try:
            json.dumps(value, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            return "unpaired surrogate escape"
    return None


def iter_raw_lines(path: str | Path, *, encoding: str = "utf-8") -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, raw)`` for every physical line, unparsed.

    Applies the same stripping, BOM, and decoding rules as :func:`iter_jsonl`:
    bytes invalid in ``encoding`` come through as U+DC80..U+DCFF (as with
    Python's ``surrogateescape``) rather than raising. Blank lines come
    through as ``""``.
    """
    with Path(path).open(encoding=encoding, errors="convmerge.surrogateescape") as f:
        for number, line in enumerate(f, 1):
            raw = line.strip()
            if number == 1:
                raw = raw.removeprefix("\ufeff").strip()
            yield number, raw


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
    A UTF-8 byte-order mark on the first line is ignored. Lines with bytes
    invalid in ``encoding``, unpaired surrogate escapes, or nesting too deep
    for the parser count as unparseable.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
    st = stats if stats is not None else ReadStats()
    bad_bytes_before = _bad_bytes_seen
    # Same loop as iter_raw_lines, inlined: this is the hot path of every command.
    with Path(path).open(encoding=encoding, errors="convmerge.surrogateescape") as f:
        for number, line in enumerate(f, 1):
            raw = line.strip()
            if number == 1:
                raw = raw.removeprefix("\ufeff").strip()
            st.lines_read += 1
            if not raw:
                st.blank += 1
                continue
            error: json.JSONDecodeError | str | None = None
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as e:
                error = e
            except RecursionError:
                error = "nested too deeply"
            else:
                check_bytes = _bad_bytes_seen != bad_bytes_before
                if check_bytes or "\\u" in raw:
                    error = _encoding_problem(
                        raw, value, check_bytes=check_bytes, encoding=encoding
                    )
            if error is not None:
                err = JsonlDecodeError(path, number, error, raw)
                if on_error == "raise":
                    raise err from None
                if on_invalid is not None:
                    on_invalid(err)
                st.invalid_json += 1
                if st.first_invalid_line is None:
                    st.first_invalid_line = number
                continue
            yield JsonlLine(number=number, raw=raw, value=value)
