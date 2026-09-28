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
- **Bracket calls** (``Team-ACE/ToolACE``): assistant turns that are only
  ``[Func Name(key="value", n=1), Other()]``, with the functions listed as a
  JSON array in the system prompt. A turn is rewritten only when every name
  is one of those functions; the system prompt is kept as it is.
- **Function-call turns** (``Locutusque/function-calling-chatml``):
  ``function-call`` turns holding Glaive-style ``{"name": ..., "arguments":
  '...'}`` and ``function-response`` turns; the function specs written into
  the system turn move to ``tools``.

Tool calls get no invented ids (like LLaMA-Factory ``function_call`` turns);
``tool`` turns carry the function ``name`` when the source gives it.
"""

from __future__ import annotations

import ast
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

    calls = function_call_value(text[len("<functioncall>") :].strip())
    return calls[0] if calls else None


def glaive_call_object(text: str) -> ToolCall | None:
    """``{"name": "f", "arguments": '{...}'}``: JSON except for the single quotes."""
    match = _GLAIVE_CALL.match("<functioncall>" + text.strip())
    return ToolCall.from_any(match.group(1), match.group(2)) if match else None


def system_tool_specs(messages: list[ChatMessage]) -> tuple[list[ChatMessage], list[Any] | None]:
    """Move function specs written into the system turn (Glaive style) to ``tools``."""
    for i, m in enumerate(messages):
        if m.role == "system" and isinstance(m.content, str) and "{" in m.content:
            text, tools = _glaive_system(m.content)
            if tools:
                return [*messages[:i], replace(m, content=text), *messages[i + 1 :]], tools
    return messages, None


# --- Bracket calls (ToolACE) --------------------------------------------------


def looks_like_bracket_calls(messages: list[ChatMessage]) -> bool:
    """Cheap test: an assistant turn that is only ``[...(...)]``."""
    for m in messages:
        if m.role == "assistant" and isinstance(m.content, str):
            text = m.content.strip()
            if text.startswith("[") and text.endswith(")]"):
                return True
    return False


def rewrite_bracket_calls(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], list[Any] | None]:
    """Turn ``[f(a=1), g()]`` assistant turns into tool calls.

    The functions come from a JSON array of ``{"name": ...}`` objects in the
    system prompt; without one, or when a name or argument does not parse,
    the turn is left as text.
    """
    tools = _system_function_list(messages)
    if not tools:
        return messages, None
    names = {t["name"] for t in tools}
    out: list[ChatMessage] = []
    for m in messages:
        calls = None
        if m.role == "assistant" and isinstance(m.content, str) and not m.tool_calls:
            calls = parse_bracket_calls(m.content, names)
        out.append(replace(m, content=None, tool_calls=tuple(calls)) if calls else m)
    return out, tools


def _system_function_list(messages: list[ChatMessage]) -> list[dict[str, Any]] | None:
    decoder = json.JSONDecoder()
    for m in messages:
        if m.role != "system" or not isinstance(m.content, str):
            continue
        text = m.content
        pos = text.find("[{")
        while pos >= 0:
            try:
                value, _ = decoder.raw_decode(text, pos)
            except ValueError:
                value = None
            if (
                isinstance(value, list)
                and value
                and all(isinstance(t, dict) and isinstance(t.get("name"), str) for t in value)
            ):
                return value
            pos = text.find("[{", pos + 2)
    return None


def parse_bracket_calls(text: str, names: set[str]) -> list[ToolCall] | None:
    """``[Name(k=v, ...), ...]`` with every name in ``names``, else ``None``."""
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    body, pos, calls = s[1:-1], 0, []
    while pos < len(body):
        while pos < len(body) and body[pos] in " \n\t,":
            pos += 1
        if pos >= len(body):
            break
        paren = body.find("(", pos)
        if paren < 0:
            return None
        name = body[pos:paren].strip()
        end = _closing_paren(body, paren)
        if name not in names or end is None:
            return None
        kwargs = _keyword_arguments(body[paren + 1 : end])
        if kwargs is None:
            return None
        calls.append(ToolCall.from_any(name, kwargs))
        pos = end + 1
    return calls or None


def _closing_paren(text: str, start: int) -> int | None:
    depth, quote, i = 0, "", start
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i if c == ")" else None
        i += 1
    return None


_JSON_NAMES = {"true": True, "false": False, "null": None}


def _keyword_arguments(args: str) -> dict[str, Any] | None:
    """``a=1, b="x"`` as a dict of literals (parsed, never evaluated)."""
    try:
        call = ast.parse(f"f({args})", mode="eval").body
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if not isinstance(call, ast.Call) or call.args:
        return None
    out: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            return None
        node = kw.value
        if isinstance(node, ast.Name) and node.id in _JSON_NAMES:
            out[kw.arg] = _JSON_NAMES[node.id]
            continue
        try:
            out[kw.arg] = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
            return None
    return out


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
