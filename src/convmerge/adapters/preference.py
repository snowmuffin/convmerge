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

Rows that score several candidate answers become a pair of the best and the
worst one (:func:`ranked_as_preference`):

- UltraFeedback: ``instruction`` plus ``completions[]`` with a ``response``
  and ``fine-grained_score`` (the mean of the four aspect ratings, which
  Argilla's cleaned binarization ranks by; the ``overall_score`` critique
  rating is used only when that is missing).
- Nectar: an HH ``prompt`` transcript plus ``answers[]`` with an ``answer``
  and a ``rank`` (1 is best).

When the best and worst scores are equal there is no pair and the row is
reported as ``no_preference``. OpenAssistant trees (see
:mod:`convmerge.adapters.oasst`) pair their best- and worst-ranked replies.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

from convmerge.adapters import oasst
from convmerge.models import TrainingExample

PREFERENCES: tuple[str, ...] = ("chosen", "rejected")

_A_WINS = ("conversation_a", "conversation_b")
_B_WINS = ("conversation_b", "conversation_a")
# Chatbot Arena ``winner`` → (chosen key, rejected key).
_ARENA_WINNERS = {"model_a": _A_WINS, "a": _A_WINS, "model_b": _B_WINS, "b": _B_WINS}

_HH_TURN = re.compile(r"\n\n(Human|Assistant):[ \t]?")
# The same transcript without the leading blank line ("Human: ...").
_HH_START = re.compile(r"\A\s*Human:[ \t]?")

# Other names for the chosen / rejected answers (distilabel math DPO,
# ``chosen_response`` / ``rejected_response``), in priority order.
PREFERENCE_ALIASES: tuple[tuple[str, str], ...] = (
    ("chosen_response", "rejected_response"),
    ("response_chosen", "response_rejected"),
    ("chosen_output", "rejected_output"),
)
_LIST_KEYS = ("conversations", "messages", "conversation")


def with_preference_keys(record: dict[str, Any]) -> dict[str, Any]:
    """``record`` with ``chosen`` / ``rejected`` taken from an alias pair
    (``chosen_response`` / ``rejected_response``, ...) or from scored
    candidates (:func:`ranked_as_preference`) when it has neither."""
    if "chosen" in record or "rejected" in record:
        return record
    if "completions" in record or "answers" in record:
        ranked = ranked_as_preference(record)
        if ranked is not None:
            return ranked
    for chosen, rejected in PREFERENCE_ALIASES:
        if chosen in record and rejected in record:
            rest = {k: v for k, v in record.items() if k not in (chosen, rejected)}
            return {**rest, "chosen": record[chosen], "rejected": record[rejected]}
    return record


def apply_preference(record: dict[str, Any], which: str) -> dict[str, Any]:
    """Return ``record`` with its ``which`` answer folded in (unchanged if absent)."""
    record = with_preference_keys(record)
    value = record.get(which)
    if value is None:
        return record
    rest = {k: v for k, v in record.items() if k not in PREFERENCES}

    if isinstance(value, str) and _is_hh(value):
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


def _is_hh(text: str) -> bool:
    if _HH_TURN.search(text):
        return True
    return bool(_HH_START.match(text)) and "Assistant:" in text


def _parse_hh(text: str) -> list[dict[str, str]]:
    # A transcript may start with "Human:" instead of "\n\nHuman:".
    parts = _HH_TURN.split(_HH_START.sub("\n\nHuman: ", text, count=1))
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
    if "chosen" in record:
        return "rejected" in record
    for chosen, rejected in PREFERENCE_ALIASES:
        if chosen in record and rejected in record:
            return True
    if "completions" in record or "answers" in record:
        return _candidates(record) is not None
    return False


# (list key, answer key, score keys in priority order, lower score is better)
_SCORED: tuple[tuple[str, str, tuple[str, ...], bool], ...] = (
    ("completions", "response", ("fine-grained_score", "overall_score"), False),  # UltraFeedback
    ("answers", "answer", ("rank",), True),  # Nectar
)
_PROMPT_KEYS = ("instruction", "prompt")


def _candidates(record: dict[str, Any]) -> list[tuple[float, str]] | None:
    """``(score, answer)`` of every scored candidate, best first; ``None`` when
    the record has no scored-candidates list."""
    for list_key, answer_key, score_keys, lower_better in _SCORED:
        items = record.get(list_key)
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            continue
        if answer_key not in items[0]:
            continue
        found: list[tuple[float, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            answer = item.get(answer_key)
            score = _score(item, score_keys)
            if isinstance(answer, str) and answer.strip() and score is not None:
                found.append((-score if lower_better else score, answer))
        if not any(_score(i, score_keys) is not None for i in items if isinstance(i, dict)):
            continue
        # Stable: among equal scores the earlier candidate wins.
        return sorted(found, key=lambda c: -c[0])
    return None


def _score(item: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def ranked_as_preference(record: dict[str, Any]) -> dict[str, Any] | None:
    """A chosen / rejected record from scored candidates (see the module doc):
    the best-scored answer against the worst-scored one. ``None`` if the
    record has no scored candidates; with fewer than two, or when the best and
    worst scores are equal, the result carries ``no_preference`` instead of
    a pair."""
    candidates = _candidates(record)
    if candidates is None:
        return None
    prompt = _first_prompt(record)
    if len(candidates) < 2 or candidates[0][0] == candidates[-1][0] or prompt is None:
        return {**record, "no_preference": True}
    rest = {k: v for k, v in record.items() if k not in ("completions", "answers", *_PROMPT_KEYS)}
    return {
        **rest,
        "prompt": prompt,
        "chosen": [_assistant(candidates[0][1])],
        "rejected": [_assistant(candidates[-1][1])],
    }


def _first_prompt(record: dict[str, Any]) -> str | list[dict[str, str]] | None:
    for key in _PROMPT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            if _is_hh(value):
                turns = _parse_hh(value)
                return turns or None
            return value
    return None


def iter_pairs(
    record: dict[str, Any], adapter: Callable[[dict[str, Any]], Iterator[TrainingExample]]
) -> Iterator[TrainingExample]:
    """Yield chosen/rejected pairs from a preference record.

    A record that is not a preference record is adapted as-is (the
    ``preference`` format then drops it as ``unrepresentable_not_preference``).
    """
    if oasst.is_tree(record):
        pair = oasst.ranked_pair(record)
        if pair is not None:
            record = {"id": record.get("message_tree_id"), "chosen": pair[0], "rejected": pair[1]}
    record = with_preference_keys(_arena_as_preference(record))
    if record.get("no_preference") is True and "chosen" not in record:
        yield TrainingExample(meta={"source": "preference"}, issues=["no_preference"])
        return
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
