"""TrainingExample → target JSON object for JSONL lines."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

from convmerge.models import ChatMessage, ContentPart, ToolCall, TrainingExample

EmitterFn = Callable[[TrainingExample], dict[str, Any]]
ToolArguments = Literal["string", "object"]

# Media part → OpenAI-style content part. ``image_url`` is the OpenAI schema;
# ``audio_url`` / ``video_url`` follow the vLLM / Qwen-VL convention for media
# given by reference.
_MEDIA_KEYS = {"image": "image_url", "audio": "audio_url", "video": "video_url"}


def emit_messages(
    example: TrainingExample,
    *,
    tool_arguments: ToolArguments = "string",
) -> dict[str, Any]:
    """OpenAI-style chat messages (one JSON object per line).

    Plain-text messages serialize exactly as ``{"role", "content"}``; optional
    keys (``name``, ``tool_calls``, ``tool_call_id``) and a top-level
    ``tools`` list appear only when present. ``tool_arguments="object"``
    writes tool-call arguments as JSON objects instead of JSON strings (some
    Hugging Face chat templates expect that).
    """
    row: dict[str, Any] = {
        "messages": [_message_dict(m, tool_arguments) for m in example.messages],
    }
    if example.tools:
        row["tools"] = example.tools
    return row


def _message_dict(m: ChatMessage, tool_arguments: ToolArguments) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role}
    if m.name is not None:
        out["name"] = m.name
    if m.content is None or isinstance(m.content, str):
        out["content"] = m.content
    else:
        out["content"] = [_part_dict(p) for p in m.content]
    if m.tool_calls:
        out["tool_calls"] = [_tool_call_dict(tc, tool_arguments) for tc in m.tool_calls]
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    return out


def _part_dict(p: ContentPart) -> dict[str, Any]:
    if p.type == "text":
        return {"type": "text", "text": p.text or ""}
    key = _MEDIA_KEYS.get(p.type)
    if key is None or p.url is None:
        # Unknown part type or unresolved placeholder: keep the bare type.
        return {"type": p.type}
    return {"type": key, key: {"url": p.url}}


def _tool_call_dict(tc: ToolCall, tool_arguments: ToolArguments) -> dict[str, Any]:
    args: Any = tc.arguments if tool_arguments == "string" else tc.arguments_object()
    out: dict[str, Any] = {}
    if tc.id is not None:
        out["id"] = tc.id
    out["type"] = "function"
    out["function"] = {"name": tc.name, "arguments": args}
    return out


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
            "instruction": msgs[0].text,
            "input": "",
            "output": msgs[1].text,
        }

    user_parts: list[str] = []
    last_asst = ""
    for m in msgs:
        if m.role == "user":
            user_parts.append(m.text)
        elif m.role == "assistant":
            last_asst = m.text
    return {
        "instruction": "\n".join(user_parts).strip(),
        "input": "",
        "output": last_asst,
    }


EMITTERS: dict[str, EmitterFn] = {
    "messages": emit_messages,
    "alpaca": emit_alpaca,
}


def get_emitter(name: str, *, tool_arguments: ToolArguments = "string") -> EmitterFn:
    if name not in EMITTERS:
        known = ", ".join(sorted(EMITTERS))
        raise ValueError(f"Unknown output format {name!r}. Choose one of: {known}")
    if tool_arguments not in ("string", "object"):
        raise ValueError(f"tool_arguments must be 'string' or 'object', got {tool_arguments!r}")
    if name == "messages" and tool_arguments != "string":
        return partial(emit_messages, tool_arguments=tool_arguments)
    return EMITTERS[name]
