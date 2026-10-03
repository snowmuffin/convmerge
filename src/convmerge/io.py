"""Shared line-by-line JSONL reader.

Every command that streams JSONL (convert, dedupe, mix, turns, filter, ...)
reads through :func:`iter_jsonl`, so blank lines, a UTF-8 BOM, and
unparseable lines are handled — and counted — the same way everywhere.
A line with bytes that are not valid in the file's encoding, an unpaired
UTF-16 surrogate escape (``"\\ud800"``), or nesting deeper than
:data:`MAX_DEPTH` is unparseable too: it is skipped like broken JSON instead
of stopping the run.
"""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import Callable, Iterable, Iterator
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


# The scanner json.loads runs (set in JSONDecoder.__init__; the stubs omit it).
_scan_once: Callable[[str, int], tuple[Any, int]] = json.JSONDecoder().scan_once  # type: ignore[attr-defined]


def _loads(raw: str) -> Any:
    """``json.loads(raw)``, calling the C scanner directly.

    Skips the Python layers of ``json.loads`` (type and BOM checks, two
    whitespace regexes), which cost about a third of the parse on typical
    rows. Anything the scanner does not accept whole (leading or trailing
    whitespace, a BOM, invalid JSON) goes through ``json.loads`` so the value
    or error is exactly the same.
    """
    try:
        value, end = _scan_once(raw, 0)
    except Exception:
        return json.loads(raw)
    if end != len(raw):
        return json.loads(raw)
    return value


# Lists and objects nested deeper than this make a line unparseable. Fixed
# rather than left to the JSON parser: before 3.14 it raised RecursionError
# near Python's recursion limit (about 1,000), 3.14 parses any depth, and the
# code that walks the value afterwards still recurses.
MAX_DEPTH = 500
# A line must be at least this long to nest deeper than MAX_DEPTH ("[" * 501
# + "]" * 501), so shorter lines skip the check without a function call.
_DEEP_LEN = 2 * MAX_DEPTH + 2


def _too_deep(raw: str, value: Any) -> bool:
    """True when ``value`` nests lists / objects more than :data:`MAX_DEPTH` deep."""
    if raw.count("[") + raw.count("{") <= MAX_DEPTH:
        return False  # cannot nest that deep: the usual case, two C scans
    stack = [(value, 1)]
    while stack:
        v, depth = stack.pop()
        if depth > MAX_DEPTH:
            return True
        items = v.values() if isinstance(v, dict) else v if isinstance(v, list) else ()
        stack.extend((x, depth + 1) for x in items if isinstance(x, (dict, list)))
    return False


def _parse_line(
    raw: str, *, check_bytes: bool, encoding: str
) -> tuple[Any, json.JSONDecodeError | str | None]:
    """``(value, None)`` for a usable stripped line, else ``(None, why)``.

    The rules of :func:`iter_jsonl`, shared with the ``--workers`` code paths
    that parse lines in other processes. ``check_bytes`` says whether the file
    had bytes invalid in ``encoding`` (only then is the line scanned for them).
    """
    try:
        value = _loads(raw)
    except json.JSONDecodeError as e:
        return None, e
    except RecursionError:
        return None, "nested too deeply"
    if len(raw) >= _DEEP_LEN and _too_deep(raw, value):
        return None, "nested too deeply"
    if check_bytes or "\\u" in raw:
        problem = _encoding_problem(raw, value, check_bytes=check_bytes, encoding=encoding)
        if problem is not None:
            return None, problem
    return value, None


class SamePathError(ValueError):
    """An output path names an input file (raised by :func:`refuse_overwrite`)."""


def refuse_overwrite(inputs: Iterable[str | Path], outputs: Iterable[str | Path | None]) -> None:
    """Raise :class:`SamePathError` when an output is an input or another output.

    Commands open their outputs before they read their input, so an output
    naming the input file would empty it. Paths that reach the same file
    through a link count as the same file.
    """
    ins = [Path(p) for p in inputs]
    outs: list[Path] = []
    for out in outputs:
        if out is None:
            continue
        o = Path(out)
        if any(_same_file(o, p) for p in ins):
            raise SamePathError(f"output {o} is the input file; write to a different path")
        if any(_same_file(o, p) for p in outs):
            raise SamePathError(f"output {o} is given twice; write each output to its own path")
        outs.append(o)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except OSError:  # one of them does not exist yet
        return a.resolve() == b.resolve()


def bad_bytes_seen() -> int:
    """How often this process has decoded an invalid byte (see :func:`iter_raw_lines`)."""
    return _bad_bytes_seen


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
    invalid in ``encoding``, unpaired surrogate escapes, or nesting deeper
    than :data:`MAX_DEPTH` count as unparseable.
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
            # _parse_line applies the same rules; inlined here for speed
            # (tests/test_io.py checks that the two agree).
            error: json.JSONDecodeError | str | None = None
            try:
                value = _loads(raw)
            except json.JSONDecodeError as e:
                error = e
            except RecursionError:
                error = "nested too deeply"
            else:
                check_bytes = _bad_bytes_seen != bad_bytes_before
                if len(raw) >= _DEEP_LEN and _too_deep(raw, value):
                    error = "nested too deeply"
                elif check_bytes or "\\u" in raw:
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
