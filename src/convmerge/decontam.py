"""Benchmark decontamination by word n-gram overlap (``convmerge decontam``).

Training rows that share a word n-gram (13 words by default, as in GPT-3)
with an evaluation set are dropped, so a model is not trained on the test it
will be scored on. Text is compared case-folded without punctuation; Han and
kana characters count as one word each, so Chinese and Japanese work without
spaces, and Korean is compared by its space-separated words.

Each evaluation row becomes one text: its string fields (or ``fields``) in
order, so a question and its answer choices form one passage. Passages
shorter than ``ngram`` words but at least ``min_tokens`` long must appear
whole; shorter ones are too generic to match on and are skipped (counted in
the report).

By default only the prompt side of each training row (system and user turns,
and a pair's shared prompt) is checked; ``check="all"`` checks answers too.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from convmerge._text import excerpt, ngrams, words
from convmerge.io import ReadStats, iter_jsonl
from convmerge.models import ChatMessage, TrainingExample

Check = Literal["prompts", "all"]
_SAMPLES = 5


@dataclass(frozen=True)
class EvalSource:
    """An evaluation set: a local JSONL file or a Hub dataset (``hf:REPO[:CONFIG[:SPLIT]]``)."""

    spec: str
    fields: tuple[str, ...] | None = None

    @property
    def name(self) -> str:
        return self.spec

    @property
    def hub(self) -> tuple[str, str | None, str] | None:
        """(repo, config, split) for ``hf:`` sources (split defaults to ``test``)."""
        if not self.spec.startswith("hf:"):
            return None
        parts = self.spec[3:].split(":")
        if not parts[0] or len(parts) > 3:
            raise ValueError(f"{self.spec}: expected hf:REPO[:CONFIG[:SPLIT]]")
        config = (parts[1] or None) if len(parts) > 1 else None
        split = parts[2] if len(parts) > 2 and parts[2] else "test"
        return parts[0], config, split


@dataclass
class DecontamStats:
    """Counters filled by :func:`decontaminate_jsonl`; ``to_report()`` gives the JSON report."""

    rows: int = 0
    kept: int = 0
    contaminated: int = 0
    unreadable: int = 0
    invalid_json: int = 0
    first_invalid_line: int | None = None
    ngram: int = 13
    check: str = "prompts"
    eval_sets: dict[str, dict[str, int]] = field(default_factory=dict)
    """Per evaluation set: ``rows`` read, ``passages`` indexed, ``too_short``
    skipped, and ``matched`` training rows."""
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_report(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "kept": self.kept,
            "contaminated": self.contaminated,
            "unreadable": self.unreadable,
            "invalid_json": self.invalid_json,
            "ngram": self.ngram,
            "check": self.check,
            "eval_sets": {k: dict(v) for k, v in self.eval_sets.items()},
            "samples": list(self.samples),
        }


class EvalIndex:
    """Hashed n-grams of evaluation passages, each mapped to its evaluation set."""

    def __init__(self, ngram: int = 13, min_tokens: int = 8) -> None:
        if ngram < 1 or min_tokens < 1:
            raise ValueError("ngram and min_tokens must be positive")
        self.ngram = ngram
        self.min_tokens = min(min_tokens, ngram)
        self._grams: dict[int, dict[int, int]] = {}
        """window length -> gram hash -> index into ``names``."""
        self.names: list[str] = []
        self.counts: list[dict[str, int]] = []

    def add(self, name: str, passages: Iterable[str]) -> dict[str, int]:
        """Index ``passages`` of the evaluation set ``name``; returns its counters."""
        source = len(self.names)
        self.names.append(name)
        counts = {"rows": 0, "passages": 0, "too_short": 0}
        self.counts.append(counts)
        n = self.ngram
        for text in passages:
            counts["rows"] += 1
            tokens = _tokens(text)
            if len(tokens) >= n:
                table = self._grams.setdefault(n, {})
                for gram in ngrams(tokens, n):
                    table.setdefault(hash(gram), source)
            elif len(tokens) >= self.min_tokens:
                self._grams.setdefault(len(tokens), {}).setdefault(hash(tuple(tokens)), source)
            else:
                counts["too_short"] += 1
                continue
            counts["passages"] += 1
        return counts

    def match(self, text: str) -> tuple[int, int] | None:
        """(evaluation set index, token offset) of the first shared n-gram, if any."""
        tokens = _tokens(text)
        for length, table in self._grams.items():
            for i, gram in enumerate(ngrams(tokens, length)):
                source = table.get(hash(gram))
                if source is not None:
                    return source, i
        return None


def _tokens(text: str) -> list[str]:
    # Single Latin letters and digits are dropped on both sides: they are mostly
    # answer-choice labels ("A.", "(b)", "1)") that a copied question may or
    # may not carry.
    return [t for t in words(text) if len(t) > 1 or not t.isascii()]


def read_eval_passages(
    source: EvalSource, *, token: str | None = None, cache_dir: str | Path | None = None
) -> Iterator[str]:
    """The passages (one per row) of an evaluation set."""
    hub = source.hub
    if hub is None:
        yield from _passages(Path(source.spec), source.fields)
        return
    from convmerge.fetch.hf import download_hf_dataset

    repo, config, split = hub
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
        path = download_hf_dataset(repo, Path(tmp) / "eval.jsonl", config=config, split=split,
                                   token=token)  # fmt: skip
        yield from _passages(path, source.fields)


def _passages(path: Path, fields: tuple[str, ...] | None) -> Iterator[str]:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation file not found: {path}")
    for line in iter_jsonl(path):
        value = line.value
        if isinstance(value, dict):
            parts = [value.get(k) for k in fields] if fields else list(value.values())
            yield "\n".join(_strings(parts))
        elif isinstance(value, str):
            yield value


def _strings(values: Iterable[Any]) -> Iterator[str]:
    for v in values:
        if isinstance(v, str):
            yield v
        elif isinstance(v, (list, tuple)):
            yield from _strings(v)
        elif isinstance(v, dict):
            yield from _strings(v.values())


def build_index(
    sources: Iterable[EvalSource],
    *,
    ngram: int = 13,
    min_tokens: int = 8,
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> EvalIndex:
    """An :class:`EvalIndex` over every evaluation set in ``sources``."""
    index = EvalIndex(ngram, min_tokens)
    for source in sources:
        index.add(source.name, read_eval_passages(source, token=token, cache_dir=cache_dir))
    return index


def decontaminate_jsonl(
    path: str | Path,
    index: EvalIndex,
    *,
    check: Check = "prompts",
    output: str | Path | None = None,
    rejects: str | Path | None = None,
    encoding: str = "utf-8",
    stats: DecontamStats | None = None,
) -> DecontamStats:
    """Find rows of ``path`` that share an n-gram with ``index``.

    With ``output``, the other rows are written there and contaminated rows
    to ``rejects`` (if given), both byte-for-byte as read.
    """
    from convmerge.adapter_resolve import resolve_adapter

    if check not in ("prompts", "all"):
        raise ValueError(f"check: expected 'prompts' or 'all', got {check!r}")
    st = stats if stats is not None else DecontamStats()
    st.ngram, st.check = index.ngram, check
    st.eval_sets = {name: {**counts, "matched": 0}
                    for name, counts in zip(index.names, index.counts)}  # fmt: skip
    adapter = resolve_adapter("auto", None, pairs=True)
    read = ReadStats()
    for target in (output, rejects):
        if target is not None:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
    out = open(output, "w", encoding="utf-8") if output is not None else None
    rej = open(rejects, "w", encoding="utf-8") if rejects is not None else None
    try:
        for line in iter_jsonl(path, encoding=encoding, stats=read):
            st.rows += 1
            examples = list(adapter(line.value)) if isinstance(line.value, dict) else []
            hit = None
            if not any(ex.messages for ex in examples):
                st.unreadable += 1  # kept: nothing to compare
            for text in _texts(examples, check):
                found = index.match(text)
                if found is not None:
                    hit = (found, text)
                    break
            if hit is None:
                st.kept += 1
                if out is not None:
                    out.write(line.raw + "\n")
                continue
            (source, offset), text = hit
            st.contaminated += 1
            st.eval_sets[index.names[source]]["matched"] += 1
            if len(st.samples) < _SAMPLES:
                st.samples.append({"line": line.number, "eval": index.names[source],
                                   "text": _around(text, offset)})  # fmt: skip
            if rej is not None:
                rej.write(line.raw + "\n")
    finally:
        for f in (out, rej):
            if f is not None:
                f.close()
        st.invalid_json = read.invalid_json
        st.first_invalid_line = read.first_invalid_line
    return st


def _texts(examples: list[TrainingExample], check: Check) -> Iterator[str]:
    for ex in examples:
        messages: list[ChatMessage] = list(ex.messages)
        if ex.rejected is not None and check == "all":
            messages += [m for m in ex.rejected if m not in ex.messages]
        for m in messages:
            if check == "all" or m.role in ("system", "user"):
                if m.text:
                    yield m.text


def _around(text: str, offset: int) -> str:
    """An excerpt of ``text`` starting near its ``offset``-th word."""
    matches = list(re.finditer(r"\S+", text))
    start = matches[min(offset, len(matches) - 1)].start() if matches else 0
    return excerpt(text, start)
