"""TrainingExample → target JSON object for JSONL lines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from convmerge.models import TrainingExample

EmitterFn = Callable[[TrainingExample], dict[str, Any]]


def emit_messages(example: TrainingExample) -> dict[str, Any]:
    """OpenAI-style chat messages (one JSON object per line)."""
    return {
        "messages": [{"role": m.role, "content": m.content} for m in example.messages],
    }


def emit_alpaca(example: TrainingExample) -> dict[str, Any]:
    """
    Alpaca instruction / input / output.

    Two-turn (user, assistant) maps cleanly. Longer conversations are flattened:
    all user contents joined into ``instruction``, last assistant into ``output``.
    """
    msgs = example.messages
    if not msgs:
        return {"instruction": "", "input": "", "output": ""}

    if len(msgs) == 2 and msgs[0].role == "user" and msgs[1].role == "assistant":
        return {
            "instruction": msgs[0].content,
            "input": "",
            "output": msgs[1].content,
        }

    user_parts: list[str] = []
    last_asst = ""
    for m in msgs:
        if m.role == "user":
            user_parts.append(m.content)
        elif m.role == "assistant":
            last_asst = m.content
    return {
        "instruction": "\n".join(user_parts).strip(),
        "input": "",
        "output": last_asst,
    }


EMITTERS: dict[str, EmitterFn] = {
    "messages": emit_messages,
    "alpaca": emit_alpaca,
}


def get_emitter(name: str) -> EmitterFn:
    if name not in EMITTERS:
        known = ", ".join(sorted(EMITTERS))
        raise ValueError(f"Unknown output format {name!r}. Choose one of: {known}")
    return EMITTERS[name]
