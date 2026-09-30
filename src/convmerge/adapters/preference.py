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
  rating breaks ties, and is used alone when ``fine-grained_score`` is
  missing).
- Nectar: an HH ``prompt`` transcript plus ``answers[]`` with an ``answer``
  and a ``rank`` (1 is best).

When the best and worst scores (tie-break included) are equal there is no
pair and the row is reported as ``no_preference``. OpenAssistant trees (see
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
    labeled = labeled_as_preference(record)
    if labeled is not None:
        return labeled
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
    return _labeled_layout(record) is not None


# (list key, answer key, score keys in priority order, lower score is better)
# (list key, answer key, score keys, tie-break keys, lower is better)
_SCORED: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...], bool], ...] = (
    ("completions", "response", ("fine-grained_score", "overall_score"), ("overall_score",),
     False),  # UltraFeedback
    ("answers", "answer", ("rank",), (), True),  # Nectar
)  # fmt: skip
_PROMPT_KEYS = ("instruction", "prompt")


def _candidates(record: dict[str, Any]) -> list[tuple[tuple[float, ...], str]] | None:
    """``(score, answer)`` of every scored candidate, best first; ``None`` when
    the record has no scored-candidates list. A score is ``(score,)``, or
    ``(score, tie-break)`` when every candidate has the tie-break score."""
    for list_key, answer_key, score_keys, tie_keys, lower_better in _SCORED:
        items = record.get(list_key)
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            continue
        if answer_key not in items[0]:
            continue
        sign = -1.0 if lower_better else 1.0
        found: list[tuple[float, float | None, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            answer = item.get(answer_key)
            score = _score(item, score_keys)
            if isinstance(answer, str) and answer.strip() and score is not None:
                tie = _score(item, tie_keys) if tie_keys else None
                found.append((sign * score, None if tie is None else sign * tie, answer))
        if not any(_score(i, score_keys) is not None for i in items if isinstance(i, dict)):
            continue
        tie_break = all(tie is not None for _, tie, _ in found)
        keyed: list[tuple[tuple[float, ...], str]] = [
            ((score, tie) if tie_break and tie is not None else (score,), answer)
            for score, tie, answer in found
        ]
        # Stable: among equal scores the earlier candidate wins.
        return sorted(keyed, key=lambda c: tuple(-x for x in c[0]))
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
    worst scores (tie-break included) are equal, the result carries
    ``no_preference`` instead of a pair."""
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


# Two answers and a label naming the better one:
# (answer A key, answer B key, label key, prompt key, label value -> 0 / 1 / None)
_LABELED: tuple[tuple[str, str, str, str, Callable[[Any], int | None]], ...] = (
    # PKU-SafeRLHF: better_response_id is the index of the better answer.
    ("response_0", "response_1", "better_response_id", "prompt",
     lambda v: v if v in (0, 1) else None),
    # SHP: labels is 1 when A is preferred, 0 when B is.
    ("human_ref_A", "human_ref_B", "labels", "history",
     lambda v: {1: 0, 0: 1}.get(v)),
    # HelpSteer3: overall_preference < 0 prefers response1, > 0 response2, 0 is a tie.
    ("response1", "response2", "overall_preference", "context",
     lambda v: 0 if v < 0 else 1 if v > 0 else None),
)  # fmt: skip


def _labeled_layout(
    record: dict[str, Any],
) -> tuple[str, str, str, str, Callable[[Any], int | None]] | None:
    for layout in _LABELED:
        if all(key in record for key in layout[:4]):
            return layout
    return None


def labeled_as_preference(record: dict[str, Any]) -> dict[str, Any] | None:
    """A chosen / rejected record from two answers and a label naming the
    better one (PKU-SafeRLHF, SHP, HelpSteer3; see ``_LABELED``). ``None`` if
    the record has none of those layouts; a tie, or a label or answer that
    cannot be read, gives ``no_preference`` instead of a pair."""
    layout = _labeled_layout(record)
    if layout is None:
        return None
    a_key, b_key, label_key, prompt_key, better = layout
    label = record[label_key]
    if isinstance(label, str) and label.strip().lstrip("-").isdigit():
        label = int(label)
    index = better(label) if isinstance(label, int) and not isinstance(label, bool) else None
    answers = (record[a_key], record[b_key])
    prompt = record[prompt_key]
    if isinstance(prompt, str) and _is_hh(prompt):
        prompt = _parse_hh(prompt) or None
    usable = all(isinstance(a, str) and a.strip() for a in answers) and (
        (isinstance(prompt, str) and prompt.strip()) or (isinstance(prompt, list) and prompt)
    )
    if index is None or not usable:
        return {**record, "no_preference": True}
    rest = {k: v for k, v in record.items() if k not in (a_key, b_key, prompt_key)}
    return {
        **rest,
        "prompt": prompt,
        "chosen": [_assistant(answers[index])],
        "rejected": [_assistant(answers[1 - index])],
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
