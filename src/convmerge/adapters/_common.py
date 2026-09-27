"""Shared record → message helpers for the chat, sharegpt, and alpaca adapters.

These cover the pieces of real SFT datasets that sit around plain
``role``/``content`` turns: OpenAI content parts and ``tool_calls``,
LLaMA-Factory ``function_call`` / ``observation`` turns and ``system`` /
``tools`` columns, and media given by reference through ``images`` /
``videos`` / ``audios`` columns (or LLaVA's single ``image``) with
``<image>``-style placeholders.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from convmerge.adapters.tool_formats import hermes_tools, rewrite_hermes, uses_hermes_tags
from convmerge.models import ChatMessage, ContentPart, ToolCall, TrainingExample

# Raw role labels whose *value* is a JSON function call (LLaMA-Factory style).
FUNCTION_CALL_ROLES: frozenset[str] = frozenset({"function_call"})

# Part ``type`` values (OpenAI, vLLM, HF chat templates) → our media type.
_PART_MEDIA_TYPES: dict[str, str] = {
    "image_url": "image",
    "image": "image",
    "input_image": "image",
    "audio_url": "audio",
    "audio": "audio",
    "video_url": "video",
    "video": "video",
}

# Record columns holding media references, and the placeholder token each uses.
_MEDIA_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("images", "image", "<image>"),
    ("image", "image", "<image>"),
    ("videos", "video", "<video>"),
    ("video", "video", "<video>"),
    ("audios", "audio", "<audio>"),
    ("audio", "audio", "<audio>"),
)

_MEDIA_COLUMN_NAMES: frozenset[str] = frozenset(c for c, _, _ in _MEDIA_COLUMNS)

MISSING = object()

# Turn keys holding an assistant's reasoning trace apart from its answer:
# ``reasoning_content`` (DeepSeek API, Qwen3 templates), ``thinking`` (gpt-oss
# templates, HuggingFaceH4/Multilingual-Thinking), ``reasoning`` (OpenRouter,
# vLLM). Only non-empty strings count.
DEFAULT_REASONING_KEYS: tuple[str, ...] = ("reasoning_content", "thinking", "reasoning")


def parse_content(value: Any, *, strip: bool = False) -> Any:
    """Return str, a tuple of :class:`ContentPart`, ``None``, or :data:`MISSING`.

    ``MISSING`` means the value is not usable message content.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() if strip else value
    if isinstance(value, list):
        parts = [p for p in (_parse_part(item, strip=strip) for item in value) if p is not None]
        return tuple(parts) if parts else MISSING
    return MISSING


def _parse_part(item: Any, *, strip: bool) -> ContentPart | None:
    if isinstance(item, str):
        text = item.strip() if strip else item
        return ContentPart("text", text=text) if text else None
    if not isinstance(item, dict):
        return None
    ptype = str(item.get("type") or "")
    if ptype in ("text", "input_text", "output_text") or (not ptype and "text" in item):
        part_text = item.get("text")
        if not isinstance(part_text, str):
            return None
        return ContentPart("text", text=part_text.strip() if strip else part_text)
    media = _PART_MEDIA_TYPES.get(ptype)
    if media is None:
        return None
    return ContentPart(media, url=_part_url(item, ptype, media))


def _part_url(item: dict[str, Any], ptype: str, media: str) -> str | None:
    for key in (ptype, media, "url", "path", "image_url"):
        v = item.get(key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            url = v.get("url") or v.get("path")
            if isinstance(url, str) and url:
                return url
    return None


def parse_tool_calls(item: dict[str, Any]) -> list[ToolCall]:
    """OpenAI ``tool_calls`` (or the legacy single ``function_call``) on a turn."""
    calls: list[ToolCall] = []
    raw = item.get("tool_calls")
    if isinstance(raw, list):
        for tc in raw:
            if not isinstance(tc, dict):
                continue
            function = tc.get("function")
            fn: dict[str, Any] = function if isinstance(function, dict) else tc
            name = fn.get("name")
            if isinstance(name, str) and name:
                tc_id = tc.get("id")
                calls.append(
                    ToolCall.from_any(
                        name, fn.get("arguments"), id=tc_id if isinstance(tc_id, str) else None
                    )
                )
    legacy = item.get("function_call")
    if not calls and isinstance(legacy, dict) and isinstance(legacy.get("name"), str):
        calls.append(ToolCall.from_any(legacy["name"], legacy.get("arguments")))
    return calls


def function_call_value(value: Any) -> list[ToolCall] | None:
    """Decode a LLaMA-Factory ``function_call`` turn value into tool calls.

    Accepts one ``{"name", "arguments"}`` object or a list of them (parallel
    calls), as a JSON string or already decoded. ``None`` if it is not one.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    items = value if isinstance(value, list) else [value]
    calls: list[ToolCall] = []
    for obj in items:
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            return None
        calls.append(ToolCall.from_any(obj["name"], obj.get("arguments")))
    return calls or None


def normalize_tools(value: Any) -> list[dict[str, Any]] | None:
    """Tool schemas in OpenAI form; accepts a JSON string and bare function specs."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return None
    tools: list[dict[str, Any]] = []
    for t in value:
        if not isinstance(t, dict):
            continue
        if isinstance(t.get("function"), dict):
            tools.append(t)
        elif isinstance(t.get("name"), str):
            tools.append({"type": "function", "function": t})
    return tools or None


def with_system(messages: list[ChatMessage], record: dict[str, Any]) -> list[ChatMessage]:
    """Prepend a top-level ``system`` (or Llama-Nemotron ``system_prompt``) column
    unless a system turn already exists."""
    system = first_text(record, ("system", "system_prompt"))
    if system is None:
        return messages
    if any(m.role == "system" for m in messages):
        return messages
    return [ChatMessage("system", system.strip()), *messages]


def attach_media(
    messages: list[ChatMessage], record: dict[str, Any]
) -> tuple[list[ChatMessage], list[str]]:
    """Bind media columns to ``<image>``-style tokens and bare media parts.

    Tokens in string content are split into text + media parts only when the
    record has the matching column, so text that merely mentions ``<image>``
    is left alone. Returns the new messages and a list of issues (unresolved
    placeholders or unused media) for validation to report.
    """
    if not any(column in record for column in _MEDIA_COLUMN_NAMES) and not any(
        isinstance(m.content, tuple) for m in messages
    ):
        return messages, []
    queues: dict[str, list[str]] = {}
    tokens: dict[str, str] = {}
    for column, media, token in _MEDIA_COLUMNS:
        refs = _media_refs(record.get(column))
        if refs:
            queues.setdefault(media, []).extend(refs)
            tokens[media] = token
    if not queues and not any(m.media for m in messages):
        return messages, []

    issues: list[str] = []
    out: list[ChatMessage] = []
    if queues and not _has_placeholder(messages, tokens):
        # No <image>-style token or media placeholder anywhere: the media
        # belongs to the conversation as a whole, so lead the first user turn
        # with it (the usual convention) rather than reporting it unused.
        messages = _prepend_media(messages, queues)
    for m in messages:
        content = m.content
        if isinstance(content, str) and tokens:
            content = _split_tokens(content, tokens)
        if content is not None and not isinstance(content, str):
            content = collapse_text(tuple(_resolve(p, queues, issues) for p in content))
        out.append(replace(m, content=content))
    for media, left in queues.items():
        if left:
            issues.append(f"unused_{media}")
    return out, issues


def collapse_text(content: Any) -> Any:
    """Turn a parts tuple with no media into a plain string (joined by newlines)."""
    if isinstance(content, tuple) and all(p.type == "text" for p in content):
        return "\n".join(p.text or "" for p in content)
    return content


def _has_placeholder(messages: list[ChatMessage], tokens: dict[str, str]) -> bool:
    for m in messages:
        if isinstance(m.content, str):
            if any(tok in m.content for tok in tokens.values()):
                return True
        elif any(p.is_media and p.url is None for p in m.media):
            return True
    return False


def _prepend_media(messages: list[ChatMessage], queues: dict[str, list[str]]) -> list[ChatMessage]:
    idx = next((i for i, m in enumerate(messages) if m.role == "user"), None)
    if idx is None:
        return messages
    m = messages[idx]
    parts = [ContentPart(media, url=ref) for media, refs in queues.items() for ref in refs]
    for refs in queues.values():
        refs.clear()
    if isinstance(m.content, str):
        rest: tuple[ContentPart, ...] = (
            (ContentPart("text", text=m.content),) if m.content.strip() else ()
        )
    else:
        rest = tuple(m.content or ())
    new = replace(m, content=(*parts, *rest))
    return [*messages[:idx], new, *messages[idx + 1 :]]


def _media_refs(value: Any) -> list[str]:
    """String references from a media column: a string, a list, or HF
    ``datasets`` ``{"path": ...}`` objects (inline bytes are not references)."""
    items = value if isinstance(value, list) else [value]
    refs: list[str] = []
    for v in items:
        if isinstance(v, dict):
            v = v.get("path") or v.get("url")
        if isinstance(v, str) and v:
            refs.append(v)
    return refs


def _split_tokens(text: str, tokens: dict[str, str]) -> str | tuple[ContentPart, ...]:
    by_token = {tok: media for media, tok in tokens.items()}
    pattern = re.compile("|".join(re.escape(t) for t in by_token))
    if not pattern.search(text):
        return text
    parts: list[ContentPart] = []
    pos = 0
    for match in pattern.finditer(text):
        _append_text(parts, text[pos : match.start()])
        parts.append(ContentPart(by_token[match.group(0)]))
        pos = match.end()
    _append_text(parts, text[pos:])
    return tuple(parts)


def _append_text(parts: list[ContentPart], text: str) -> None:
    text = text.strip()
    if text:
        parts.append(ContentPart("text", text=text))


def _resolve(part: ContentPart, queues: dict[str, list[str]], issues: list[str]) -> ContentPart:
    if not part.is_media or part.url is not None:
        return part
    queue = queues.get(part.type)
    if queue:
        return ContentPart(part.type, url=queue.pop(0))
    issues.append(f"unresolved_{part.type}")
    return part


def first_role(item: dict[str, Any], role_keys: Iterable[str]) -> str | None:
    for rk in role_keys:
        v = item.get(rk)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return None


def build_example(
    msgs: list[ChatMessage], record: dict[str, Any], *, meta: dict[str, object]
) -> TrainingExample:
    """Wrap turns in a :class:`TrainingExample`, attaching the record-level
    ``system`` / ``tools`` / media columns and decoding Hermes-style
    ``<tool_call>`` / ``<tool_response>`` tags (see
    :mod:`convmerge.adapters.tool_formats`)."""
    msgs = with_system(msgs, record)
    tools = record.get("tools")
    if uses_hermes_tags(msgs, record):
        msgs = rewrite_hermes(msgs)
        if tools is None:
            tools = hermes_tools(msgs)
    msgs, issues = attach_media(msgs, record)
    return TrainingExample(
        messages=msgs,
        meta=source_meta(record, meta),
        tools=normalize_tools(tools),
        issues=issues,
    )


def coerce_messages(
    convs: list[Any],
    *,
    role_keys: tuple[str, ...],
    content_keys: tuple[str, ...],
    role_map: dict[str, str],
    strip: bool = False,
    reasoning_keys: tuple[str, ...] = DEFAULT_REASONING_KEYS,
) -> list[ChatMessage]:
    """Map a list of turn dicts to messages.

    Roles come from the first of ``role_keys`` present (lower-cased, then
    ``role_map``); content from the first usable ``content_keys`` value
    (string or parts). ``function_call`` turns become assistant tool calls.
    Turns without content (missing, null, or blank text) are skipped unless
    they carry tool calls; with ``strip=True`` text is also stripped. An
    assistant turn's reasoning comes from the first non-empty string under
    ``reasoning_keys``.
    """
    out: list[ChatMessage] = []
    rk0, ck0 = role_keys[0], content_keys[0]
    for item in convs:
        if not isinstance(item, dict):
            continue
        if len(item) == 2:
            # Fast path for the common plain turn: exactly {role, content}
            # strings under the primary keys. Same result as the general path.
            role_v, content_v = item.get(rk0), item.get(ck0)
            if type(role_v) is str and type(content_v) is str:
                role = role_v.strip().lower()
                if role and role not in FUNCTION_CALL_ROLES:
                    if content_v.strip():
                        text = content_v.strip() if strip else content_v
                        out.append(ChatMessage(role_map.get(role, role), text))
                    continue
        role_raw = first_role(item, role_keys)
        if role_raw is None:
            continue
        content = _first_content(item, content_keys, strip=strip)

        if role_raw in FUNCTION_CALL_ROLES:
            calls = function_call_value(content)
            if calls:
                out.append(ChatMessage("assistant", None, tool_calls=calls))
                continue

        tool_calls = parse_tool_calls(item)
        empty = (
            content is MISSING
            or content is None
            or (isinstance(content, str) and not content.strip())
        )
        if empty and not tool_calls:
            continue
        name = item.get("name")
        tool_call_id = item.get("tool_call_id")
        role = role_map.get(role_raw, role_raw)
        out.append(
            ChatMessage(
                role,
                None if content is MISSING else collapse_text(content),
                tool_calls=tool_calls,
                tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
                name=name if isinstance(name, str) and name else None,
                reasoning=first_text(item, reasoning_keys) if role == "assistant" else None,
            )
        )
    return out


def first_text(record: dict[str, Any], keys: Iterable[str]) -> str | None:
    """The first non-blank string under ``keys``."""
    for k in keys:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _first_content(item: dict[str, Any], content_keys: tuple[str, ...], *, strip: bool) -> Any:
    """First usable content under ``content_keys``; ``None`` if only nulls were found."""
    found_null = False
    for ck in content_keys:
        if ck not in item:
            continue
        parsed = parse_content(item[ck], strip=strip)
        if parsed is None:
            found_null = True
        elif parsed is not MISSING:
            return parsed
    return None if found_null else MISSING


def source_meta(record: dict[str, Any], meta: dict[str, object]) -> dict[str, object]:
    """Add the record's own ``id`` (string or integer) to adapter metadata."""
    rid = record.get("id")
    if isinstance(rid, (str, int)) and not isinstance(rid, bool):
        return {**meta, "id": rid}
    return meta
