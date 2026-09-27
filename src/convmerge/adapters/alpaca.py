"""Alpaca-style instruction / input / output → TrainingExample."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from convmerge.adapters._common import attach_media, source_meta, with_system
from convmerge.models import ChatMessage, TrainingExample


def iter_from_alpaca_line(
    record: dict[str, Any], *, reasoning: str | None = None
) -> Iterator[TrainingExample]:
    """
    One JSON object per line: instruction, optional input, output.

    Maps to a single user message + single assistant message. The
    LLaMA-Factory extensions are honored too: ``history`` (a list of
    ``[user, assistant]`` pairs) becomes earlier turns, ``system`` becomes a
    system message, and ``images`` / ``videos`` / ``audios`` columns bind to
    ``<image>``-style tokens in the text. ``reasoning`` is the answer's
    reasoning trace, when the caller found one in the record.
    """
    instruction = _text(record.get("instruction"))
    inp = _text(record.get("input"))
    output = _text(record.get("output")) or _text(record.get("response"))

    user_parts = [instruction]
    if inp:
        user_parts.append(inp)
    user_content = "\n".join(user_parts).strip()

    if not user_content and not output:
        return

    messages: list[ChatMessage] = _history(record.get("history"))
    if user_content:
        messages.append(ChatMessage(role="user", content=user_content))
    if output:
        messages.append(ChatMessage(role="assistant", content=output, reasoning=reasoning))

    messages = with_system(messages, record)
    messages, issues = attach_media(messages, record)
    yield TrainingExample(
        messages=messages, meta=source_meta(record, {"source": "alpaca"}), issues=issues
    )


def _text(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _history(value: Any) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    if not isinstance(value, list):
        return out
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        q, a = _text(pair[0]), _text(pair[1])
        if q and a:
            out.append(ChatMessage("user", q))
            out.append(ChatMessage("assistant", a))
    return out
