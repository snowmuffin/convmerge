"""Reasoning traces: kept in a field, or inline as ``<think>...</think>``.

Datasets store an assistant turn's reasoning in one of two ways: in a field of
its own (``reasoning_content``, read by Qwen3 / DeepSeek chat templates, or
``thinking``, read by gpt-oss), or inline at the start of the answer as
``<think>...</think>`` (DeepSeek-R1 distillations, OpenR1, most ShareGPT
reasoning sets). :class:`~convmerge.models.ChatMessage` keeps the first in
``reasoning`` and leaves the second in ``content``; the helpers here move a
trace between the two or remove it.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Literal

from convmerge.models import ChatMessage, ContentPart

Content = str | tuple[ContentPart, ...]

ReasoningMode = Literal["keep", "inline", "reasoning_content", "thinking", "drop"]
REASONING_MODES: tuple[str, ...] = ("keep", "inline", "reasoning_content", "thinking", "drop")

# A leading <think> block (and the blank lines after it).
_LEADING_THINK = re.compile(r"\A\s*<think>(.*?)</think>[ \t]*(?:\r?\n)*", re.S)


def split_inline(text: str) -> tuple[str | None, str]:
    """``(reasoning, answer)`` of text that starts with a ``<think>`` block.

    ``(None, text)`` when it does not (or the block is never closed).
    """
    match = _LEADING_THINK.match(text)
    if match is None:
        return None, text
    return match.group(1).strip("\n"), text[match.end() :]


def join_inline(reasoning: str, answer: str) -> str:
    """``<think>\\nreasoning\\n</think>\\n\\nanswer`` (the Qwen3 / DeepSeek-R1 layout)."""
    return f"<think>\n{reasoning.strip(chr(10))}\n</think>\n\n{answer.lstrip(chr(10))}"


def reasoning_text(m: ChatMessage) -> str | None:
    """The turn's reasoning, from the field or an inline ``<think>`` block."""
    if m.reasoning:
        return m.reasoning
    text = _leading_text(m)
    return split_inline(text)[0] if text is not None else None


def has_reasoning(m: ChatMessage) -> bool:
    return bool(reasoning_text(m))


def to_field(m: ChatMessage) -> ChatMessage:
    """Move an inline ``<think>`` block into ``reasoning`` (no-op if there is none)."""
    if m.role != "assistant":
        return m
    text = _leading_text(m)
    if text is None:
        return m
    found, answer = split_inline(text)
    if found is None:
        return m
    reasoning = f"{m.reasoning}\n\n{found}" if m.reasoning else found
    return replace(m, content=_with_leading_text(m, answer), reasoning=reasoning or None)


def to_inline(m: ChatMessage) -> ChatMessage:
    """Write ``reasoning`` into the content as a leading ``<think>`` block."""
    if not m.reasoning:
        return m
    text = _leading_text(m)
    if isinstance(m.content, tuple) and text is None:
        content: Content = (ContentPart("text", text=join_inline(m.reasoning, "")), *m.content)
    else:
        content = _with_leading_text(m, join_inline(m.reasoning, text or ""))
    return replace(m, content=content, reasoning=None)


def strip_reasoning(m: ChatMessage) -> ChatMessage:
    """Remove the turn's reasoning, both the field and an inline block."""
    if m.role != "assistant":
        return m
    m = to_field(m)
    return replace(m, reasoning=None) if m.reasoning is not None else m


def _leading_text(m: ChatMessage) -> str | None:
    content = m.content
    if isinstance(content, str):
        return content
    if isinstance(content, tuple) and content and content[0].type == "text":
        return content[0].text or ""
    return None


def _with_leading_text(m: ChatMessage, text: str) -> Content:
    content = m.content
    if isinstance(content, tuple) and content:
        return (ContentPart("text", text=text), *content[1:])
    return text
