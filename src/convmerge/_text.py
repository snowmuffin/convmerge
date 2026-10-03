"""Text helpers shared by the quality commands (``filter``, ``decontam``, near dedupe),
and for counts in messages printed to people."""

from __future__ import annotations

import re
from collections.abc import Iterator

# Han ideographs and kana are written without spaces, so each is its own
# token; every other run of letters or digits (Latin, Hangul, ...) is a word.
_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿"
_TOKEN = re.compile(rf"[{_CJK}]|[^\W{_CJK}]+")
_FENCED = re.compile(r"```.*?(?:```|\Z)", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_SPACE = re.compile(r"\s+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def words(text: str) -> list[str]:
    """Lower-cased word tokens (one per Han or kana character), punctuation dropped."""
    return _TOKEN.findall(text.casefold())


def ngrams(tokens: list[str], n: int) -> Iterator[tuple[str, ...]]:
    """Consecutive ``n``-token windows of ``tokens``."""
    return zip(*(tokens[i:] for i in range(n)))


def fold(text: str, *, casefold: bool = True) -> str:
    """Case-folded text with typographic quotes made plain and whitespace collapsed."""
    text = text.translate(_QUOTES)
    return " ".join((text.casefold() if casefold else text).split())


def without_code(text: str) -> str:
    """``text`` with fenced and inline code removed."""
    return _INLINE_CODE.sub(" ", _FENCED.sub(" ", text))


def excerpt(text: str, start: int = 0, width: int = 160) -> str:
    """About ``width`` characters of ``text`` around ``start``, on one line."""
    begin = max(0, start - width // 4)
    piece = _SPACE.sub(" ", text[begin : begin + width]).strip()
    return ("..." if begin else "") + piece + ("..." if begin + width < len(text) else "")


def count(n: int, singular: str, plural: str | None = None) -> str:
    """``"1 row"`` / ``"2 rows"`` (``plural`` defaults to ``singular + "s"``)."""
    return f"{n:,} {singular if n == 1 else plural or singular + 's'}"


def agree(n: int, singular: str, plural: str) -> str:
    """The verb form that agrees with ``n``: ``agree(1, "is", "are")`` is ``"is"``."""
    return singular if n == 1 else plural
