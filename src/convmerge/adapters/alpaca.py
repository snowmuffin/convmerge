"""Alpaca-style instruction / input / output → TrainingExample."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from convmerge.models import ChatMessage, TrainingExample


def iter_from_alpaca_line(record: dict[str, Any]) -> Iterator[TrainingExample]:
    """
    One JSON object per line: instruction, optional input, output.

    Maps to a single user message + single assistant message.
    """
    instruction = (record.get("instruction") or "").strip()
    inp = (record.get("input") or "").strip()
    output = (record.get("output") or "").strip() or (record.get("response") or "").strip()

    user_parts = [instruction]
    if inp:
        user_parts.append(inp)
    user_content = "\n".join(user_parts).strip()

    if not user_content and not output:
        return

    messages: list[ChatMessage] = []
    if user_content:
        messages.append(ChatMessage(role="user", content=user_content))
    if output:
        messages.append(ChatMessage(role="assistant", content=output))

    if not messages:
        return

    yield TrainingExample(messages=messages, meta={"source": "alpaca"})
