"""Tool-calling datasets that do not use OpenAI ``tool_calls``.

Each encoding is turned into the standard model — assistant turns with
:class:`~convmerge.models.ToolCall` objects, ``tool`` turns for the results,
and OpenAI-form ``tools`` schemas — so every output format and chat template
sees ordinary tool calls:

- **Hermes** (NousResearch ``hermes-function-calling-v1`` and the many
  datasets in its format): calls as ``<tool_call>{"name", "arguments"}
  </tool_call>`` blocks inside assistant text, results as
  ``<tool_response>`` blocks inside ``tool`` turns, schemas in a ``tools``
  column or a ``<tools>[...]</tools>`` block of the system prompt. Applied to
  any conversation (``messages`` / ``conversations``) that shows those tags.
- **Glaive** (``glaiveai/glaive-function-calling-v2``): a ``system`` string
  listing the functions as JSON objects and a ``chat`` transcript of
  ``USER:`` / ``ASSISTANT:`` / ``FUNCTION RESPONSE:`` turns, with calls as
  ``<functioncall> {...}``.
- **xLAM** (``Salesforce/xlam-function-calling-60k``): ``query``, ``answers``
  (the calls, as a JSON string) and ``tools`` (JSON string).

Tool calls get no invented ids (like LLaMA-Factory ``function_call`` turns);
``tool`` turns carry the function ``name`` when the source gives it.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from convmerge.models import ChatMessage, ToolCall

# --- Hermes ------------------------------------------------------------------

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_TOOLS_BLOCK = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)


def uses_hermes_tags(messages: list[ChatMessage], record: dict[str, Any]) -> bool:
    """Cheap test run on every conversation: is there anything to rewrite?"""
    for m in messages:
        if m.role == "tool" or (m.role == "system" and "<tools>" in m.text):
            return True
    return "tools" in record and any(
        m.role == "assistant" and "<tool_call>" in m.text for m in messages
    )


def rewrite_hermes(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Turn ``<tool_call>`` / ``<tool_response>`` text into structured turns.

    Messages without those tags are returned unchanged. A block that is not
    valid JSON with a ``name`` is left in the text, so nothing is lost.
    """
    out: list[ChatMessage] = []
    for m in messages:
        if m.role == "assistant" and isinstance(m.content, str) and "<tool_call>" in m.content:
            out.append(_hermes_calls(m))
        elif m.role == "tool" and isinstance(m.content, str) and "<tool_response>" in m.content:
            out.extend(_hermes_responses(m))
        else:
            out.append(m)
    return out


def hermes_tools(messages: list[ChatMessage]) -> list[Any] | None:
    """Tool schemas from a ``<tools>[...]</tools>`` block in the system prompt."""
    for m in messages:
        if m.role != "system":
            continue
        # The prompt usually mentions "<tools> </tools> XML tags" before the
        # real block, so try every match.
        for match in _TOOLS_BLOCK.finditer(m.text):
            value = _json(match.group(1))
            if isinstance(value, (list, dict)):
                return value if isinstance(value, list) else [value]
    return None


def _hermes_calls(m: ChatMessage) -> ChatMessage:
    text = m.content if isinstance(m.content, str) else ""
    calls: list[ToolCall] = []

    def take(match: re.Match[str]) -> str:
        obj = _json(match.group(1))
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            calls.append(ToolCall.from_any(obj["name"], obj.get("arguments")))
            return ""
        return match.group(0)

    rest = _TOOL_CALL.sub(take, text).strip()
    if not calls:
        return m
    return replace(m, content=rest or None, tool_calls=(*m.tool_calls, *calls))


def _hermes_responses(m: ChatMessage) -> list[ChatMessage]:
    text = m.content if isinstance(m.content, str) else ""
    out: list[ChatMessage] = []
    for block in _TOOL_RESPONSE.findall(text):
        obj = _json(block)
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and "content" in obj:
            content = obj["content"]
            body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            out.append(ChatMessage("tool", body, name=obj["name"]))
        else:
            out.append(ChatMessage("tool", block, name=m.name))
    return out or [m]


# --- Glaive ------------------------------------------------------------------

_GLAIVE_TURN = re.compile(r"(?:^|\n)\s*(USER|ASSISTANT|FUNCTION RESPONSE):[ \t]*")
_END = "<|endoftext|>"


def is_glaive(record: dict[str, Any]) -> bool:
    chat = record.get("chat")
    return isinstance(chat, str) and chat.lstrip().startswith(("USER:", "SYSTEM:"))


def glaive_messages(record: dict[str, Any]) -> tuple[list[ChatMessage], list[Any] | None]:
    """Messages and tool schemas from a Glaive ``system`` + ``chat`` record."""
    system, tools = _glaive_system(record.get("system"))
    messages: list[ChatMessage] = [ChatMessage("system", system)] if system else []
    parts = _GLAIVE_TURN.split(str(record.get("chat") or ""))
    last_call: str | None = None
    # parts: [preamble, speaker, text, speaker, text, ...]
    for i in range(1, len(parts) - 1, 2):
        speaker, text = parts[i], parts[i + 1].replace(_END, "").strip()
        if speaker == "USER":
            if text:
                messages.append(ChatMessage("user", text))
        elif speaker == "ASSISTANT":
            call = _glaive_call(text)
            if call is not None:
                last_call = call.name
                messages.append(ChatMessage("assistant", None, tool_calls=(call,)))
            elif text:
                messages.append(ChatMessage("assistant", text))
        elif text:
            messages.append(ChatMessage("tool", text, name=last_call))
    return messages, tools


def _glaive_system(value: Any) -> tuple[str, list[Any] | None]:
    if not isinstance(value, str):
        return "", None
    text = value.strip()
    if text.startswith("SYSTEM:"):
        text = text[len("SYSTEM:") :].strip()
    start = text.find("{")
    if start < 0:
        return text, None
    decoder = json.JSONDecoder()
    tools: list[Any] = []
    pos = start
    while pos < len(text):
        try:
            obj, end = decoder.raw_decode(text, pos)
        except ValueError:
            break
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            tools.append(obj)
        pos = end
        while pos < len(text) and text[pos].isspace():
            pos += 1
    if not tools:
        return text, None
    # Keep the instruction sentence; the schemas move to ``tools``.
    return text[:start].strip().rstrip("-").strip(), tools


_GLAIVE_CALL = re.compile(
    r"<functioncall>\s*\{\s*\"name\"\s*:\s*\"([^\"]+)\"\s*,\s*\"arguments\"\s*:\s*'(.*)'\s*\}\s*$",
    re.DOTALL,
)


def _glaive_call(text: str) -> ToolCall | None:
    if not text.startswith("<functioncall>"):
        return None
    from convmerge.adapters._common import function_call_value

    match = _GLAIVE_CALL.match(text)
    if match:  # arguments quoted as a JSON string inside single quotes
        return ToolCall.from_any(match.group(1), match.group(2))
    calls = function_call_value(text[len("<functioncall>") :].strip())
    return calls[0] if calls else None


# --- xLAM --------------------------------------------------------------------


def is_xlam(record: dict[str, Any]) -> bool:
    return isinstance(record.get("query"), str) and "answers" in record


def xlam_messages(record: dict[str, Any]) -> list[ChatMessage]:
    from convmerge.adapters._common import function_call_value

    calls = function_call_value(record.get("answers"))
    messages = [ChatMessage("user", str(record["query"]))]
    if calls:
        messages.append(ChatMessage("assistant", None, tool_calls=calls))
    return messages


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None
