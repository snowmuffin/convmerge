"""Single-turn <-> multi-turn record conversion helpers."""

from __future__ import annotations

from typing import Any


def single_turn_to_multi_turn_record(
    data: dict[str, Any],
    *,
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
) -> dict[str, Any] | None:
    """Turn ``{instruction, input, output}`` into a two-turn messages sample.

    Returns ``None`` when either instruction or output is missing/empty so that
    partial rows are silently dropped rather than producing garbage messages.
    """
    instr = data.get(instruction_key, "")
    inp = data.get(input_key, "")
    out = data.get(output_key, "")

    if not isinstance(instr, str) or not instr.strip():
        return None
    if not isinstance(out, str) or not out.strip():
        return None

    if isinstance(inp, str) and inp.strip():
        user_content = instr.strip() + "\n\n" + inp.strip()
    else:
        user_content = instr.strip()

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": out.strip()},
        ]
    }


def multi_turn_to_single_turn_record(
    data: dict[str, Any],
    *,
    joiner: str = "\n",
) -> dict[str, Any] | None:
    """Flatten a messages sample into ``{instruction, input, output}``.

    All user messages are joined with ``joiner`` into ``instruction``; the last
    assistant message becomes ``output``. Returns ``None`` when no assistant
    turn exists.
    """
    msgs = data.get("messages") or []
    if not isinstance(msgs, list) or not msgs:
        return None

    user_parts: list[str] = []
    last_assistant: str | None = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        if role == "user":
            user_parts.append(content)
        elif role == "assistant":
            last_assistant = content

    if last_assistant is None:
        return None

    return {
        "instruction": joiner.join(user_parts).strip(),
        "input": "",
        "output": last_assistant.strip(),
    }
