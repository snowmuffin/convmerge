"""Label rows with a model's answer about them (``convmerge label``).

Each row's conversation is rendered as plain text and sent to an
OpenAI-compatible endpoint through :mod:`convmerge.llm` (cache, host policy,
retries, budget). The answer is parsed strictly and written to
``meta[NAME]``; everything else in the row is kept. Design:
``docs/design/labeling.md``; usage: ``docs/label.md``.

Two tasks:

- **classes**: the answer must be one of the class names (case-insensitive,
  the first non-empty line, trimmed of whitespace, quotes and a final
  period); the label is the class name as given.
- **scale** ``(lo, hi)``: the first non-empty line must be one integer in
  the range; the label is that integer.

An answer that does not parse is asked again once, quoting it and restating
the format; a second failure writes ``null`` and is counted. A row is never
dropped, and rows no adapter can read are written as they are. Labels are
not replaced unless ``overwrite`` is set: a row that already has
``meta[NAME]`` stops the run before any call.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

from convmerge.io import refuse_overwrite

__all__ = ["LabelStats", "label_jsonl"]

DEFAULT_MAX_CHARS = 8000
_BATCH = 256
_SAMPLES = 3
_SAMPLE_CHARS = 160

_SYSTEM = (
    "You label rows of a training dataset. Answer with exactly what is asked, "
    "on one line, and nothing else."
)
_CLASSES_PROMPT = """$question

Classes: $classes

Conversation:
<<<
$conversation
>>>

Answer with exactly one class name from the list and nothing else."""
_SCALE_PROMPT = """Rate the conversation below on a scale of $scale.

$rubric

Conversation:
<<<
$conversation
>>>

Answer with one integer from $scale and nothing else."""
_QUOTES = "\"'`*“”‘’"


@dataclass
class LabelStats:
    """Counts for one :func:`label_jsonl` call."""

    name: str = ""
    task: str = ""
    """``classes`` or ``scale``."""
    host: str = ""
    model: str = ""
    prompt_sha256: str = ""
    """SHA-256 of the prompt template (built-in or ``prompt``)."""
    params: dict[str, Any] = field(default_factory=dict)
    rows: int = 0
    labeled: int = 0
    truncated: int = 0
    """Rows whose rendered conversation was cut to ``max_chars``."""
    unreadable: int = 0
    """Rows written without a label: invalid JSON, no conversation, or a non-object meta."""
    parse_failures: int = 0
    call_failures: int = 0
    """Rows with no answer: every retry failed, or (``cache_only``) not in the cache."""
    values: dict[str, int] = field(default_factory=dict)
    """Count per class (in class order) or per scale point."""
    failed_samples: list[str] = field(default_factory=list)
    """Up to three answers that did not parse."""
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    served_models: list[str] = field(default_factory=list)
    fingerprints: list[str] = field(default_factory=list)

    def to_report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "host": self.host,
            "model": self.model,
            "served_models": self.served_models,
            "fingerprints": self.fingerprints,
            "prompt_sha256": self.prompt_sha256,
            "params": self.params,
            "rows": self.rows,
            "labeled": self.labeled,
            "truncated": self.truncated,
            "unreadable": self.unreadable,
            "parse_failures": self.parse_failures,
            "call_failures": self.call_failures,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "values": self.values,
            "failed_samples": self.failed_samples,
        }


def parse_scale(text: str) -> tuple[int, int]:
    """``"1-5"`` -> ``(1, 5)``; the low end must be below the high end."""
    m = re.fullmatch(r"\s*(-?\d+)\s*-\s*(-?\d+)\s*", text)
    if not m or int(m.group(1)) >= int(m.group(2)):
        raise ValueError(f"scale: expected LO-HI with LO < HI (for example 1-5), got {text!r}")
    return int(m.group(1)), int(m.group(2))


def label_jsonl(
    src: str | Path,
    dst: str | Path,
    *,
    endpoint: str,
    model: str,
    name: str,
    classes: Sequence[str] | None = None,
    scale: tuple[int, int] | str | None = None,
    question: str | None = None,
    rubric: str | None = None,
    prompt: str | None = None,
    meta_key: str = "meta",
    overwrite: bool = False,
    max_chars: int = DEFAULT_MAX_CHARS,
    cache: str | Path | None = None,
    cache_only: bool = False,
    api_key_env: str | None = None,
    allow_external: bool = False,
    workers: int = 4,
    timeout: float = 60.0,
    max_calls: int | None = None,
    encoding: str = "utf-8",
    stats: LabelStats | None = None,
) -> LabelStats:
    """Write ``src``'s rows to ``dst`` with ``meta[name]`` set; see the module docstring.

    ``rubric`` and ``prompt`` are text (the CLI reads them from files);
    ``prompt`` is a :class:`string.Template` for the user turn with
    ``$conversation``, ``$classes``, ``$scale``, ``$question`` and ``$rubric``.
    ``cache`` is a SQLite file (``None``: no cache). Raises ``ValueError``
    for bad options or existing labels, and :class:`convmerge.llm.LLMError`
    (or its ``BudgetExceeded``) when the run must stop; ``dst`` is then not
    written, and the answers so far stay in the cache.
    """
    from convmerge.llm import CallStats, Client, Endpoint, ResponseCache

    task = _Task.build(classes, scale, question, rubric, prompt)
    if not name:
        raise ValueError("label: a name is required")
    if not meta_key:
        raise ValueError("label: meta_key must not be empty")
    if max_chars < 100:
        raise ValueError(f"label: max_chars must be at least 100, got {max_chars}")
    refuse_overwrite([src], [dst])
    if not overwrite:
        _refuse_existing(src, meta_key, name, encoding)

    ep = Endpoint(endpoint, model, api_key_env=api_key_env, allow_external=allow_external,
                  timeout=timeout, max_tokens=task.max_tokens)  # fmt: skip
    st = stats if stats is not None else LabelStats()
    st.name, st.task, st.host, st.model = name, task.kind, ep.host, model
    st.prompt_sha256 = hashlib.sha256(task.template.encode("utf-8")).hexdigest()
    st.params = ep.params()
    st.values = {str(v): 0 for v in task.points}
    calls = CallStats()
    store = ResponseCache(cache) if cache is not None else None
    client = Client(ep, cache=store, cache_only=cache_only, max_calls=max_calls,
                    workers=workers, stats=calls)  # fmt: skip
    try:
        _run(src, dst, task, client, st, meta_key, name, max_chars, encoding)
    finally:
        st.calls, st.cache_hits, st.retries = calls.calls, calls.cache_hits, calls.retries
        st.served_models, st.fingerprints = sorted(calls.models), sorted(calls.fingerprints)
        if store is not None:
            store.close()
    return st


@dataclass
class _Task:
    kind: str
    template: str
    fields: dict[str, str]
    points: list[Any]
    """The values a label can take: class names, or the scale's integers."""
    max_tokens: int

    @classmethod
    def build(cls, classes: Sequence[str] | None, scale: tuple[int, int] | str | None,
              question: str | None, rubric: str | None, prompt: str | None) -> _Task:  # fmt: skip
        if (classes is None) == (scale is None):
            raise ValueError("label: give exactly one of classes or scale")
        question = (question or "").strip()
        rubric = (rubric or "").strip()
        if classes is not None:
            names = [str(c).strip() for c in classes]
            if len(names) < 2 or any(not c for c in names):
                raise ValueError("label: classes needs at least two non-empty names")
            folded = [c.casefold() for c in names]
            if len(set(folded)) != len(folded):
                raise ValueError(f"label: class names repeat (case-insensitive): {names}")
            if prompt is None and not question:
                raise ValueError("label: classes needs a question (or a prompt template)")
            fields = {"classes": ", ".join(names), "scale": "", "question": question,
                      "rubric": rubric}  # fmt: skip
            task = cls("classes", prompt if prompt is not None else _CLASSES_PROMPT, fields,
                       names, max(16, max(len(c) for c in names) + 8))  # fmt: skip
        else:
            lo, hi = parse_scale(scale) if isinstance(scale, str) else scale  # type: ignore[misc]
            if lo >= hi:
                raise ValueError(f"label: scale needs LO < HI, got {lo}-{hi}")
            if prompt is None and not (rubric or question):
                raise ValueError("label: scale needs a rubric or a question (or a prompt template)")
            fields = {"classes": "", "scale": f"{lo}-{hi}", "question": question,
                      "rubric": rubric or question}  # fmt: skip
            task = cls("scale", prompt if prompt is not None else _SCALE_PROMPT, fields,
                       list(range(lo, hi + 1)), 16)  # fmt: skip
        if "$conversation" not in task.template and "${conversation}" not in task.template:
            raise ValueError("label: the prompt template has no $conversation placeholder")
        task.user("")  # a bad placeholder fails here, before any row is read
        return task

    def user(self, conversation: str) -> str:
        try:
            return Template(self.template).substitute(self.fields, conversation=conversation)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"label: the prompt template has an unknown or malformed placeholder ({e}); "
                "use $conversation, $classes, $scale, $question, $rubric, and $$ for a "
                "literal $"
            ) from None

    def parse(self, answer: str) -> Any:
        line = next((ln.strip() for ln in answer.splitlines() if ln.strip()), "")
        line = line.strip(_QUOTES + " \t").removesuffix(".").strip(_QUOTES + " \t")
        if self.kind == "classes":
            by_fold = {c.casefold(): c for c in self.points}
            return by_fold.get(line.casefold())
        if re.fullmatch(r"[+-]?\d+", line) and int(line) in self.points:
            return int(line)
        return None

    def restate(self) -> str:
        if self.kind == "classes":
            return "Answer with exactly one of these class names and nothing else: " + \
                ", ".join(self.points)  # fmt: skip
        return f"Answer with one integer from {self.fields['scale']} and nothing else."


def _refuse_existing(src: str | Path, meta_key: str, name: str, encoding: str) -> None:
    found, first = 0, None
    for number, value in _rows(src, encoding):
        meta = value.get(meta_key) if isinstance(value, dict) else None
        if isinstance(meta, dict) and name in meta:
            found += 1
            first = first or number
    if found:
        raise ValueError(
            f"{found:,} row(s) already have {meta_key}.{name} (first on line {first}); "
            "pass --overwrite (overwrite=True) to replace them"
        )


def _rows(src: str | Path, encoding: str) -> Any:
    """``(line number, parsed value or the raw line)`` for every non-blank line."""
    with open(src, encoding=encoding, newline="") as f:
        for number, raw in enumerate(f, 1):
            line = raw.rstrip("\r\n")
            if number == 1:
                line = line.removeprefix("﻿")
            if not line.strip():
                continue
            try:
                yield number, json.loads(line)
            except json.JSONDecodeError:
                yield number, _Raw(line)


@dataclass(frozen=True)
class _Raw:
    line: str


def _run(src: str | Path, dst: str | Path, task: _Task, client: Any, st: LabelStats,
         meta_key: str, name: str, max_chars: int, encoding: str) -> None:  # fmt: skip
    from convmerge._output import atomic_text_writer
    from convmerge.adapter_resolve import resolve_adapter

    adapter = resolve_adapter("auto", None, pairs=True)
    with atomic_text_writer(dst) as out:
        batch: list[tuple[Any, str | None]] = []
        for _, value in _rows(src, encoding):
            st.rows += 1
            text = _conversation(value, adapter, meta_key)
            if text is None:
                st.unreadable += 1
            batch.append((value, text))
            if len(batch) >= _BATCH:
                _label_batch(batch, task, client, st, meta_key, name, max_chars, out)
                batch = []
        _label_batch(batch, task, client, st, meta_key, name, max_chars, out)


def _label_batch(batch: list[tuple[Any, str | None]], task: _Task, client: Any, st: LabelStats,
                 meta_key: str, name: str, max_chars: int, out: Any) -> None:  # fmt: skip
    todo = [i for i, (_, text) in enumerate(batch) if text is not None]
    users: dict[int, str] = {}
    for i in todo:
        text = batch[i][1] or ""
        if len(text) > max_chars:
            st.truncated += 1
            text = _cut_middle(text, max_chars)
        users[i] = task.user(text)
    answers = client.complete_many([_messages(users[i]) for i in todo])
    labels: dict[int, Any] = {}
    again: list[int] = []
    first: dict[int, str] = {}
    for i, answer in zip(todo, answers):
        if answer is None:
            st.call_failures += 1
            continue
        label = task.parse(answer.text)
        if label is None:
            again.append(i)
            first[i] = answer.text
        else:
            labels[i] = label
    if again:
        retry = [_messages(users[i] + "\n\nYour previous answer was:\n<<<\n" + first[i] +
                           "\n>>>\nThat is not in the required format. " + task.restate())
                 for i in again]  # fmt: skip
        for i, answer in zip(again, client.complete_many(retry)):
            label = task.parse(answer.text) if answer is not None else None
            if label is not None:
                labels[i] = label
                continue
            if answer is None:
                st.call_failures += 1
            else:
                st.parse_failures += 1
                if len(st.failed_samples) < _SAMPLES:
                    st.failed_samples.append(answer.text[:_SAMPLE_CHARS])
    for i, (value, conversation) in enumerate(batch):
        if isinstance(value, _Raw):
            out.write(value.line + "\n")
            continue
        if conversation is not None:
            label = labels.get(i)
            if value.get(meta_key) is None:
                value[meta_key] = {}
            value[meta_key][name] = label
            if label is not None:
                st.labeled += 1
                st.values[str(label)] += 1
        out.write(json.dumps(value, ensure_ascii=False) + "\n")


def _messages(user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def _conversation(value: Any, adapter: Any, meta_key: str) -> str | None:
    """The row as the model sees it, or ``None`` (counted unreadable) when it has none."""
    if not isinstance(value, dict) or not isinstance(value.get(meta_key) or {}, dict):
        return None
    try:
        examples = [ex for ex in adapter(value) if ex.messages]
    except Exception:  # noqa: BLE001 - an adapter's failure on one row is that row's problem
        examples = []
    if not examples:
        return None
    # A preference row's messages are its prompt and chosen answer.
    return "\n\n".join(_turn(m) for m in examples[0].messages)


def _turn(m: Any) -> str:
    parts: list[str] = []
    if isinstance(m.content, str):
        parts.append(m.content)
    elif m.content:
        for p in m.content:
            if p.is_media:
                parts.append(f"[{p.type}]")
            elif p.text:
                parts.append(p.text)
    for call in m.tool_calls:
        parts.append(f"[tool call] {call.name}({call.arguments})")
    return f"{m.role}: " + "\n".join(parts)


def _cut_middle(text: str, limit: int) -> str:
    """Keep the start and the end of ``text``, at most ``limit`` characters in all."""
    keep = limit - len(f"\n[... {len(text):,} characters cut ...]\n")
    head = keep // 2
    marker = f"\n[... {len(text) - keep:,} characters cut ...]\n"
    return text[:head] + marker + text[len(text) - (keep - head) :]
