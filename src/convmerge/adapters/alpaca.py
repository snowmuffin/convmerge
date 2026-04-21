"""Alpaca-style instruction / input / output → TrainingExample."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from convmerge.models import ChatMessage, TrainingExample

DEFAULT_INSTRUCTION_KEYS: tuple[str, ...] = ("instruction", "question", "prompt")
DEFAULT_OUTPUT_KEYS: tuple[str, ...] = ("output", "response", "answer", "solution")
DEFAULT_INPUT_KEYS: tuple[str, ...] = ("input", "context")


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


def remap_to_alpaca(
    record: dict[str, Any],
    *,
    instruction_keys: Iterable[str] = DEFAULT_INSTRUCTION_KEYS,
    output_keys: Iterable[str] = DEFAULT_OUTPUT_KEYS,
    input_keys: Iterable[str] = DEFAULT_INPUT_KEYS,
    drop_keys: Iterable[str] = (),
) -> dict[str, Any] | None:
    """Pull an alpaca-shaped record out of a messy single-turn row.

    For each slot (instruction / output / input) the first key from the
    corresponding priority list that is present on ``record`` wins. ``drop_keys``
    are ignored (useful for per-source ids / urls that you don't want to
    carry downstream).

    Returns a fresh ``{"instruction", "input", "output"}`` dict, or ``None``
    when no usable instruction/output pair can be assembled.
    """
    obj = dict(record)
    for k in drop_keys:
        obj.pop(k, None)

    instr = _first_nonempty_str(obj, instruction_keys)
    output = _first_nonempty_str(obj, output_keys)
    if instr is None or output is None:
        return None

    inp = _first_nonempty_str(obj, input_keys) or ""
    return {
        "instruction": instr,
        "input": inp,
        "output": output,
    }


def _first_nonempty_str(record: dict[str, Any], keys: Iterable[str]) -> str | None:
    for k in keys:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None
