"""ShareGPT-style conversations (from/value) → TrainingExample."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

from convmerge.models import ChatMessage, TrainingExample

# Common ShareGPT role labels
_FROM_TO_ROLE: dict[str, str] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "bing": "assistant",
}

TURN_MODES: tuple[str, ...] = ("pairs", "full")

PAIRS_DEFAULT_WARNING = (
    "sharegpt adapter: split a conversation into independent user/assistant pairs, "
    "dropping its system prompt and/or earlier turns. The default turn_mode changes "
    "from 'pairs' to 'full' (keep the whole conversation) in convmerge 0.6.0. Set it "
    'explicitly to silence this warning, e.g. --adapter-kwargs \'{"sharegpt": '
    '{"turn_mode": "full"}}\' (or "pairs" to keep the current behavior), or '
    "adapter_options.sharegpt.turn_mode in a preset."
)


def _normalize_role(from_key: str) -> str:
    return _FROM_TO_ROLE.get(from_key.lower().strip(), from_key.lower().strip())


def _text(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def iter_from_sharegpt_line(
    record: dict[str, Any],
    *,
    turn_mode: str | None = None,
) -> Iterator[TrainingExample]:
    """
    One JSON object with ``conversations``: list of ``{"from": ..., "value": ...}``.

    ``turn_mode`` selects how a multi-turn conversation is emitted:

    - ``"full"``: one example holding the whole conversation (system prompt
      and every turn, in order). Turns with an empty value are dropped; a
      conversation needs at least one user and one assistant turn.
    - ``"pairs"``: one example per consecutive human→assistant pair (the
      pre-0.6 behavior). System prompts, unpaired turns, and the context of
      earlier turns are dropped.
    - ``None`` (default): ``"pairs"``, but emits a :class:`FutureWarning`
      whenever the result differs from ``"full"``; the default becomes
      ``"full"`` in 0.6.0.
    """
    if turn_mode not in (None, *TURN_MODES):
        raise ValueError(f"Unknown sharegpt turn_mode {turn_mode!r}. Use 'pairs' or 'full'.")

    convs = record.get("conversations")
    if not isinstance(convs, list):
        return

    if turn_mode == "full":
        full = _full_messages(convs)
        if full is not None:
            yield TrainingExample(messages=full, meta={"source": "sharegpt"})
        return

    pairs = _pair_messages(convs)
    if turn_mode is None and _pairs_lose_information(pairs, _full_messages(convs)):
        warnings.warn(PAIRS_DEFAULT_WARNING, FutureWarning, stacklevel=2)
    for messages in pairs:
        yield TrainingExample(messages=messages, meta={"source": "sharegpt"})


def _pair_messages(convs: list[Any]) -> list[list[ChatMessage]]:
    out: list[list[ChatMessage]] = []
    i = 0
    while i + 1 < len(convs):
        a, b = convs[i], convs[i + 1]
        if not isinstance(a, dict) or not isinstance(b, dict):
            i += 1
            continue
        ra = _normalize_role(str(a.get("from", "")))
        rb = _normalize_role(str(b.get("from", "")))
        if ra == "user" and rb == "assistant":
            out.append(
                [
                    ChatMessage(role="user", content=_text(a.get("value"))),
                    ChatMessage(role="assistant", content=_text(b.get("value"))),
                ]
            )
            i += 2
        else:
            i += 1
    return out


def _full_messages(convs: list[Any]) -> list[ChatMessage] | None:
    messages: list[ChatMessage] = []
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = _normalize_role(str(turn.get("from", "")))
        content = _text(turn.get("value"))
        if not role or not content:
            continue
        messages.append(ChatMessage(role=role, content=content))
    roles = {m.role for m in messages}
    if "user" not in roles or "assistant" not in roles:
        return None
    return messages


def _pairs_lose_information(pairs: list[list[ChatMessage]], full: list[ChatMessage] | None) -> bool:
    if full is None:
        return False
    return not (len(pairs) == 1 and pairs[0] == full)
