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
"""

from __future__ import annotations

import re
from typing import Any

PREFERENCES: tuple[str, ...] = ("chosen", "rejected")

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
