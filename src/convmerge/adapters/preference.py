"""Turn a preference record (chosen / rejected) into a plain SFT record.

Preference datasets store the answer to train on apart from the prompt. With
``preference="chosen"`` (CLI ``--preference chosen``) the chosen answer is
folded back into the conversation before the normal adapter runs, so DPO /
reward-model data can be reused for SFT. Shapes handled:

- LLaMA-Factory ranking: ``conversations`` / ``messages`` (or alpaca
  ``instruction``) plus ``chosen`` as one turn (``{"from", "value"}`` or
  ``{"role", "content"}``) or a string.
- HH-RLHF: ``chosen`` is a whole ``"\\n\\nHuman: ...\\n\\nAssistant: ..."``
  transcript.
- UltraFeedback-binarized / TRL: ``chosen`` is a message list — the whole
  conversation, or only the assistant continuation of a ``prompt`` (string
  or message list).

For the ``preference`` output format, :func:`iter_pairs` adapts the record
twice — once with each answer folded in — and returns one example holding
both conversations (``messages`` = chosen, ``rejected`` = rejected). Chatbot
Arena rows (``conversation_a`` / ``conversation_b`` with a ``winner``) are
read as a pair too.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

from convmerge.models import TrainingExample

PREFERENCES: tuple[str, ...] = ("chosen", "rejected")

_A_WINS = ("conversation_a", "conversation_b")
_B_WINS = ("conversation_b", "conversation_a")
# Chatbot Arena ``winner`` → (chosen key, rejected key).
_ARENA_WINNERS = {"model_a": _A_WINS, "a": _A_WINS, "model_b": _B_WINS, "b": _B_WINS}

_HH_TURN = re.compile(r"\n\n(Human|Assistant):[ \t]?")
_LIST_KEYS = ("conversations", "messages", "conversation")


def apply_preference(record: dict[str, Any], which: str) -> dict[str, Any]:
    """Return ``record`` with its ``which`` answer folded in (unchanged if absent)."""
    value = record.get(which)
    if value is None:
        return record
    rest = {k: v for k, v in record.items() if k not in PREFERENCES}

    if isinstance(value, str) and _HH_TURN.search(value):
        turns = _parse_hh(value)
        if turns:
            return {**_without_lists(rest), "messages": turns}

    if isinstance(value, list):
        if any(_role(m) == "user" for m in value):
            return {**_without_lists(rest), "messages": value}
        prompt = rest.pop("prompt", None)
        if isinstance(prompt, list):
            return {**_without_lists(rest), "messages": [*prompt, *value]}
        if isinstance(prompt, str):
            user = {"role": "user", "content": prompt}
            return {**_without_lists(rest), "messages": [user, *value]}
        key = _list_key(rest)
        if key is not None:
            return {**rest, key: [*rest[key], *value]}
        return record

    key = _list_key(rest)
    if key is not None:
        turns = rest[key]
        if isinstance(value, dict):
            return {**rest, key: [*turns, value]}
        if isinstance(value, str):
            uses_from = any(isinstance(t, dict) and "from" in t for t in turns)
            turn = {"from": "gpt", "value": value} if uses_from else _assistant(value)
            return {**rest, key: [*turns, turn]}
        return record

    if isinstance(value, str):
        return {**rest, "output": value}
    if isinstance(value, dict):
        text = value.get("value", value.get("content"))
        if isinstance(text, str):
            return {**rest, "output": text}
    return record


def _parse_hh(text: str) -> list[dict[str, str]]:
    parts = _HH_TURN.split(text)
    # parts: [preamble, speaker, text, speaker, text, ...]
    turns: list[dict[str, str]] = []
    for i in range(1, len(parts) - 1, 2):
        role = "user" if parts[i] == "Human" else "assistant"
        content = parts[i + 1].strip()
        if content:
            turns.append({"role": role, "content": content})
    return turns


def _list_key(record: dict[str, Any]) -> str | None:
    for key in _LIST_KEYS:
        if isinstance(record.get(key), list):
            return key
    return None


def _without_lists(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in _LIST_KEYS}


def _role(m: Any) -> str | None:
    if not isinstance(m, dict):
        return None
    role = m.get("role", m.get("from"))
    if not isinstance(role, str):
        return None
    return {"human": "user", "gpt": "assistant"}.get(role.lower(), role.lower())


def _assistant(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


def is_preference_record(record: dict[str, Any]) -> bool:
    return "chosen" in record and "rejected" in record


def iter_pairs(
    record: dict[str, Any], adapter: Callable[[dict[str, Any]], Iterator[TrainingExample]]
) -> Iterator[TrainingExample]:
    """Yield chosen/rejected pairs from a preference record.

    A record that is not a preference record is adapted as-is (the
    ``preference`` format then drops it as ``unrepresentable_not_preference``).
    """
    record = _arena_as_preference(record)
    if not is_preference_record(record):
        yield from adapter(record)
        return
    chosen = list(adapter(apply_preference(record, "chosen")))
    rejected = list(adapter(apply_preference(record, "rejected")))
    if len(chosen) != 1 or len(rejected) != 1:
        # Nothing to pair up; the chosen side (if any) is reported as unpaired.
        yield from chosen
        return
    c, r = chosen[0], rejected[0]
    yield TrainingExample(
        messages=c.messages,
        meta=c.meta,
        tools=c.tools or r.tools,
        issues=list(dict.fromkeys([*c.issues, *r.issues])),
        rejected=r.messages,
    )


def _arena_as_preference(record: dict[str, Any]) -> dict[str, Any]:
    if "conversation_a" not in record or "conversation_b" not in record:
        return record
    keys = _ARENA_WINNERS.get(str(record.get("winner") or "").strip().lower())
    if keys is None:
        return record  # a tie or no winner: not a pair
    rest = {k: v for k, v in record.items() if k not in ("conversation_a", "conversation_b")}
    return {**rest, "chosen": record[keys[0]], "rejected": record[keys[1]]}
