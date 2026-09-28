"""TrainingExample → target JSON object for JSONL lines."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal

from convmerge.models import ChatMessage, ContentPart, ToolCall, TrainingExample
from convmerge.plugins import EMITTER_GROUP, load_entry_points, unknown_name_message
from convmerge.reasoning import REASONING_MODES, ReasoningMode, strip_reasoning, to_field, to_inline

EmitterFn = Callable[[TrainingExample], dict[str, Any]]


class UnrepresentableExample(ValueError):
    """Raised by an emitter for an example its format cannot hold losslessly.

    ``convert`` drops the example and counts ``reason`` (an
    ``unrepresentable_*`` code) in its stats.
    """

    def __init__(self, reason: str, message: str = ""):
        super().__init__(message or reason)
        self.reason = reason


ToolArguments = Literal["string", "object"]
AlpacaMultiturn = Literal["flatten", "history", "drop"]
ToolContent = Literal["empty", "null"]


@dataclass(frozen=True)
class EmitOptions:
    """Output-format options shared by the emitters.

    - ``tool_arguments``: ``messages`` writes tool-call arguments as a JSON
      string (``"string"``, OpenAI style) or as an object (``"object"``).
    - ``keep_meta``: also write the example's provenance (``source``, source
      ``id``, pairwise ``branch``) under ``meta_key``. ``True`` keeps every
      key; a sequence keeps only those keys. Off by default.
    - ``alpaca_multiturn``: how ``alpaca`` handles anything other than one
      user→assistant pair. ``"flatten"`` (default) joins user turns into
      ``instruction`` and keeps the last assistant turn (lossy; counted);
      ``"history"`` writes earlier pairs to a LLaMA-Factory ``history``
      list (lossless for alternating conversations); ``"drop"`` drops them.
    - ``reasoning``: where an assistant turn's reasoning goes. ``"keep"``
      (default) leaves inline ``<think>`` blocks as they are and writes a
      separate trace as ``reasoning_content``; ``"reasoning_content"`` /
      ``"thinking"`` move inline blocks into that field (the one the target
      chat template reads: Qwen3 / DeepSeek, gpt-oss); ``"inline"`` writes
      every trace as a leading ``<think>`` block; ``"drop"`` removes them.
      Formats without a reasoning field (``alpaca``, ``sharegpt``) always
      write traces inline.
    - ``tool_content``: the ``content`` of an assistant turn that only calls
      tools. ``"empty"`` (default) writes ``""``, which every common chat
      template accepts; ``"null"`` writes ``null`` (several templates,
      including Qwen3's and gpt-oss's, fail on it).
    - ``meta_values``: constant fields written under ``meta_key`` on every
      row (with or without ``keep_meta``), e.g. ``{"dataset": "kullm",
      "license": "apache-2.0"}`` to keep each row's origin after a mix.
    """

    tool_arguments: ToolArguments = "string"
    keep_meta: bool | Sequence[str] = False
    meta_key: str = "meta"
    alpaca_multiturn: AlpacaMultiturn = "flatten"
    reasoning: ReasoningMode = "keep"
    tool_content: ToolContent = "empty"
    meta_values: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.tool_arguments not in ("string", "object"):
            raise ValueError(
                f"tool_arguments must be 'string' or 'object', got {self.tool_arguments!r}"
            )
        if self.alpaca_multiturn not in ("flatten", "history", "drop"):
            raise ValueError(
                "alpaca_multiturn must be 'flatten', 'history', or 'drop', "
                f"got {self.alpaca_multiturn!r}"
            )
        if self.reasoning not in REASONING_MODES:
            raise ValueError(
                f"reasoning must be one of {', '.join(REASONING_MODES)}, got {self.reasoning!r}"
            )
        if self.tool_content not in ("empty", "null"):
            raise ValueError(f"tool_content must be 'empty' or 'null', got {self.tool_content!r}")
        if not isinstance(self.keep_meta, bool):
            object.__setattr__(self, "keep_meta", tuple(self.keep_meta))
        if self.meta_values is not None:
            values = dict(self.meta_values)
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in values.items()):
                raise ValueError("meta_values must map strings to strings")
            object.__setattr__(self, "meta_values", values)


# Media part → OpenAI-style content part. ``image_url`` is the OpenAI schema;
# ``audio_url`` / ``video_url`` follow the vLLM / Qwen-VL convention for media
# given by reference.
_MEDIA_KEYS = {"image": "image_url", "audio": "audio_url", "video": "video_url"}


def emit_messages(
    example: TrainingExample,
    *,
    tool_arguments: ToolArguments = "string",
    options: EmitOptions | None = None,
) -> dict[str, Any]:
    """OpenAI-style chat messages (one JSON object per line).

    Plain-text messages serialize exactly as ``{"role", "content"}``; optional
    keys (``name``, ``tool_calls``, ``tool_call_id``) and a top-level
    ``tools`` list appear only when present. ``tool_arguments="object"``
    writes tool-call arguments as JSON objects instead of JSON strings (some
    Hugging Face chat templates expect that).
    """
    opts = options if options is not None else EmitOptions(tool_arguments=tool_arguments)
    row: dict[str, Any] = {"messages": _message_dicts(example.messages, opts)}
    if example.tools:
        row["tools"] = example.tools
    return _with_meta(row, example, options)


def _with_meta(
    row: dict[str, Any], example: TrainingExample, options: EmitOptions | None
) -> dict[str, Any]:
    if options is None or (options.keep_meta is False and not options.meta_values):
        return row
    meta: dict[str, Any] = {}
    if options.keep_meta is True:
        meta = dict(example.meta)
    elif options.keep_meta:
        meta = {k: v for k, v in example.meta.items() if k in options.keep_meta}
    meta.update(options.meta_values or {})
    if meta:
        row[options.meta_key] = meta
    return row


def _message_dicts(msgs: Sequence[ChatMessage], opts: EmitOptions) -> list[dict[str, Any]]:
    key = "thinking" if opts.reasoning == "thinking" else "reasoning_content"
    return [
        _message_dict(
            place_reasoning(m, opts.reasoning),
            opts.tool_arguments,
            reasoning_key=key,
            tool_content=opts.tool_content,
        )
        for m in msgs
    ]


def place_reasoning(m: ChatMessage, mode: ReasoningMode, *, has_field: bool = True) -> ChatMessage:
    """``m`` with its reasoning where ``mode`` puts it (see :class:`EmitOptions`).

    ``has_field=False`` is for formats that cannot store a separate trace:
    anything kept is written inline.
    """
    if m.role != "assistant":
        return m
    if mode == "drop":
        return strip_reasoning(m)
    if mode == "inline" or not has_field:
        return to_inline(m)
    if mode in ("reasoning_content", "thinking"):
        return to_field(m)
    return m


def _inline_reasoning(example: TrainingExample, opts: EmitOptions) -> TrainingExample:
    """For formats without a reasoning field: traces inline (or dropped)."""
    sides = [example.messages, *([example.rejected] if example.rejected else [])]
    if opts.reasoning != "drop" and not any(m.reasoning for side in sides for m in side):
        return example

    def place(msgs: list[ChatMessage]) -> list[ChatMessage]:
        return [place_reasoning(m, opts.reasoning, has_field=False) for m in msgs]

    rejected = place(example.rejected) if example.rejected is not None else None
    return replace(example, messages=place(example.messages), rejected=rejected)


def _message_dict(
    m: ChatMessage,
    tool_arguments: ToolArguments,
    *,
    reasoning_key: str = "reasoning_content",
    tool_content: ToolContent = "empty",
) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role}
    if m.name is not None:
        out["name"] = m.name
    if m.content is None:
        out["content"] = "" if m.tool_calls and tool_content == "empty" else None
    elif isinstance(m.content, str):
        out["content"] = m.content
    else:
        out["content"] = [_part_dict(p) for p in m.content]
    if m.reasoning is not None:
        out[reasoning_key] = m.reasoning
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


def emit_alpaca(
    example: TrainingExample,
    *,
    options: EmitOptions | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Alpaca instruction / input / output (plus ``system`` / ``history``).

    One user→assistant pair maps cleanly; system turns go to a ``system``
    field (LLaMA-Factory style). Longer conversations follow
    ``options.alpaca_multiturn`` (see :class:`EmitOptions`); ``"flatten"``
    appends ``"lossy_multiturn_flattened"`` to ``notes`` so ``convert`` can
    count it. Tool calls and media cannot be written in this format: such
    examples raise :class:`UnrepresentableExample`.
    """
    opts = options or EmitOptions()
    msgs = _inline_reasoning(example, opts).messages
    if example.tools or any(m.tool_calls or m.role == "tool" for m in msgs):
        raise UnrepresentableExample("unrepresentable_tool_calls")
    if any(m.media for m in msgs):
        raise UnrepresentableExample("unrepresentable_media")
    if not msgs:
        return {"instruction": "", "input": "", "output": ""}

    system = "\n".join(m.text for m in msgs if m.role == "system" and m.text)
    turns = [m for m in msgs if m.role != "system"]
    if len(turns) == 2 and turns[0].role == "user" and turns[1].role == "assistant":
        row: dict[str, Any] = {"instruction": turns[0].text, "input": "", "output": turns[1].text}
    elif opts.alpaca_multiturn == "drop":
        raise UnrepresentableExample("unrepresentable_multiturn")
    elif opts.alpaca_multiturn == "history":
        row = _alpaca_with_history(turns)
    else:
        row = _alpaca_flattened(turns)
        if notes is not None:
            notes.append("lossy_multiturn_flattened")
    if system:
        row["system"] = system
    return _with_meta(row, example, options)


def _alpaca_flattened(turns: list[ChatMessage]) -> dict[str, Any]:
    user_parts: list[str] = []
    last_asst = ""
    for m in turns:
        if m.role == "user":
            user_parts.append(m.text)
        elif m.role == "assistant":
            last_asst = m.text
    return {"instruction": "\n".join(user_parts).strip(), "input": "", "output": last_asst}


def _alpaca_with_history(turns: list[ChatMessage]) -> dict[str, Any]:
    roles = [m.role for m in turns]
    if len(turns) < 2 or len(turns) % 2 or roles != ["user", "assistant"] * (len(turns) // 2):
        # Only strictly alternating user/assistant conversations fit history.
        raise UnrepresentableExample("unrepresentable_multiturn")
    pairs = [[turns[i].text, turns[i + 1].text] for i in range(0, len(turns), 2)]
    last_user, last_asst = pairs.pop()
    return {"instruction": last_user, "input": "", "output": last_asst, "history": pairs}


def emit_preference(
    example: TrainingExample, *, options: EmitOptions | None = None
) -> dict[str, Any]:
    """Preference pairs for DPO-style training (TRL conversational format).

    ``{"prompt": [...], "chosen": [...], "rejected": [...]}``: the turns the
    chosen and rejected conversations share are the prompt, and each side
    keeps its own continuation, which must start with an assistant turn.
    ``tools`` is added when present. Examples that are not pairs, or whose
    two sides are identical or have no answer after the prompt, raise
    :class:`UnrepresentableExample`.
    """
    opts = options or EmitOptions()
    prompt, chosen_tail, rejected_tail = split_pair(example)
    row: dict[str, Any] = {
        "prompt": _message_dicts(prompt, opts),
        "chosen": _message_dicts(chosen_tail, opts),
        "rejected": _message_dicts(rejected_tail, opts),
    }
    if example.tools:
        row["tools"] = example.tools
    return _with_meta(row, example, options)


def split_pair(
    example: TrainingExample,
) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage]]:
    """(prompt, chosen continuation, rejected continuation) of a preference example.

    Raises :class:`UnrepresentableExample` when the example is not a usable pair.
    """
    chosen, rejected = example.messages, example.rejected
    if rejected is None:
        raise UnrepresentableExample("unrepresentable_not_preference")
    n = 0
    while n < len(chosen) and n < len(rejected) and chosen[n] == rejected[n]:
        n += 1
    prompt, chosen_tail, rejected_tail = chosen[:n], chosen[n:], rejected[n:]
    if not chosen_tail and not rejected_tail:
        raise UnrepresentableExample("unrepresentable_identical_pair")
    if (
        not chosen_tail
        or not rejected_tail
        or chosen_tail[0].role != "assistant"
        or rejected_tail[0].role != "assistant"
        or not any(m.role == "user" for m in prompt)
        or not _has_answer(rejected_tail)
    ):
        raise UnrepresentableExample("unrepresentable_incomplete_pair")
    return prompt, chosen_tail, rejected_tail


# Asks the adapter to keep both answers of preference records (see
# :func:`convmerge.adapters.preference.iter_pairs`).
emit_preference.preference_pairs = True  # type: ignore[attr-defined]


def _has_answer(turns: list[ChatMessage]) -> bool:
    return any(m.role == "assistant" and (m.text.strip() or m.tool_calls) for m in turns)


def wants_pairs(output_format: str) -> bool:
    """Whether ``output_format`` consumes chosen/rejected pairs."""
    fn = EMITTERS.get(output_format)
    return bool(getattr(fn, "preference_pairs", False))


# --- LLaMA-Factory ShareGPT ---------------------------------------------------

_MEDIA_TOKENS = {"image": "<image>", "video": "<video>", "audio": "<audio>"}
_MEDIA_COLUMNS = {"image": "images", "video": "videos", "audio": "audios"}


def emit_sharegpt(
    example: TrainingExample,
    *,
    options: EmitOptions | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """LLaMA-Factory / Unsloth ShareGPT rows.

    ``{"conversations": [{"from": "human" | "gpt" | "function_call" |
    "observation", "value": ...}], "system": ..., "tools": "<JSON>",
    "images": [...]}``. LLaMA-Factory needs turns that alternate user-side
    (``human`` / ``observation``) and model-side (``gpt`` / ``function_call``),
    starting with ``human`` and ending with a model turn, and the system
    prompt in its own column. Tool calls become one ``function_call`` turn
    (a list for parallel calls); consecutive tool results one ``observation``.
    Media parts become ``<image>``-style tokens plus ``images`` / ``videos`` /
    ``audios`` columns. Text next to a tool call cannot be kept (counted as
    ``lossy_tool_call_text``); other conversations that do not fit raise
    :class:`UnrepresentableExample`.
    """
    example = _inline_reasoning(example, options or EmitOptions())
    system, turns = _system_and_turns(example.messages)
    media: dict[str, list[str]] = {}
    conversations = _sharegpt_turns(turns, media, notes)
    if not conversations or conversations[-1]["from"] not in ("gpt", "function_call"):
        raise UnrepresentableExample("unrepresentable_role_order")
    row: dict[str, Any] = {"conversations": conversations}
    _sharegpt_extras(row, system, example.tools, media)
    return _with_meta(row, example, options)


def emit_sharegpt_preference(
    example: TrainingExample, *, options: EmitOptions | None = None
) -> dict[str, Any]:
    """LLaMA-Factory ranking rows (DPO / ORPO / reward modeling).

    ``{"conversations": <prompt turns>, "chosen": {"from": "gpt", "value"},
    "rejected": {"from": "gpt", "value"}, "system", "tools"}``. Each side
    must be a single assistant answer (LLaMA-Factory's ranking format has no
    room for multi-turn or tool-call continuations).
    """
    example = _inline_reasoning(example, options or EmitOptions())
    prompt, chosen_tail, rejected_tail = split_pair(example)
    if not (_single_answer(chosen_tail) and _single_answer(rejected_tail)):
        raise UnrepresentableExample("unrepresentable_pair_continuation")
    system, turns = _system_and_turns(prompt)
    media: dict[str, list[str]] = {}
    conversations = _sharegpt_turns(turns, media, None)
    if not conversations or conversations[-1]["from"] not in ("human", "observation"):
        raise UnrepresentableExample("unrepresentable_role_order")
    row: dict[str, Any] = {
        "conversations": conversations,
        "chosen": {"from": "gpt", "value": chosen_tail[0].text},
        "rejected": {"from": "gpt", "value": rejected_tail[0].text},
    }
    _sharegpt_extras(row, system, example.tools, media)
    return _with_meta(row, example, options)


emit_sharegpt_preference.preference_pairs = True  # type: ignore[attr-defined]


def _single_answer(tail: list[ChatMessage]) -> bool:
    return len(tail) == 1 and not tail[0].tool_calls and not tail[0].media and bool(tail[0].text)


def _system_and_turns(msgs: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
    """Leading system messages joined, and the rest (a later system turn does not fit)."""
    i = 0
    while i < len(msgs) and msgs[i].role == "system":
        i += 1
    if any(m.role == "system" for m in msgs[i:]):
        raise UnrepresentableExample("unrepresentable_role_order")
    system = "\n".join(m.text for m in msgs[:i] if m.text)
    return system, msgs[i:]


def _sharegpt_turns(
    turns: list[ChatMessage], media: dict[str, list[str]], notes: list[str] | None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(turns):
        m = turns[i]
        if m.role == "user":
            entry = {"from": "human", "value": _with_media_tokens(m, media)}
        elif m.role == "assistant" and m.tool_calls:
            if m.text.strip() and notes is not None:
                notes.append("lossy_tool_call_text")
            calls = [{"name": tc.name, "arguments": tc.arguments_object()} for tc in m.tool_calls]
            value = calls[0] if len(calls) == 1 else calls
            entry = {"from": "function_call", "value": json.dumps(value, ensure_ascii=False)}
        elif m.role == "assistant":
            entry = {"from": "gpt", "value": _with_media_tokens(m, media)}
        elif m.role == "tool":
            results = []
            while i < len(turns) and turns[i].role == "tool":
                results.append(turns[i].text)
                i += 1
            out.append({"from": "observation", "value": "\n".join(results)})
            continue
        else:
            raise UnrepresentableExample("unrepresentable_role_order")
        out.append(entry)
        i += 1
    user_side = ("human", "observation")
    for pos, entry in enumerate(out):
        if (entry["from"] in user_side) != (pos % 2 == 0):
            raise UnrepresentableExample("unrepresentable_role_order")
    return out


def _with_media_tokens(m: ChatMessage, media: dict[str, list[str]]) -> str:
    if m.content is None or isinstance(m.content, str):
        return m.text
    parts: list[str] = []
    for p in m.content:
        if p.type == "text":
            parts.append(p.text or "")
        elif p.type in _MEDIA_TOKENS and p.url is not None:
            parts.append(_MEDIA_TOKENS[p.type])
            media.setdefault(_MEDIA_COLUMNS[p.type], []).append(p.url)
        else:
            raise UnrepresentableExample("unrepresentable_media")
    return "\n".join(x for x in parts if x)


def _sharegpt_extras(
    row: dict[str, Any],
    system: str,
    tools: list[dict[str, Any]] | None,
    media: dict[str, list[str]],
) -> None:
    if system:
        row["system"] = system
    if tools:
        # LLaMA-Factory takes the tools column as a JSON string of function specs.
        specs = [t["function"] if isinstance(t.get("function"), dict) else t for t in tools]
        row["tools"] = json.dumps(specs, ensure_ascii=False)
    row.update(media)


EMITTERS: dict[str, EmitterFn] = {
    "messages": emit_messages,
    "alpaca": emit_alpaca,
    "preference": emit_preference,
    "sharegpt": emit_sharegpt,
    "sharegpt-preference": emit_sharegpt_preference,
}
BUILTIN_FORMATS = frozenset(EMITTERS)


def get_emitter(
    name: str,
    *,
    tool_arguments: ToolArguments | None = None,
    options: EmitOptions | None = None,
    notes: list[str] | None = None,
) -> EmitterFn:
    """Return the emitter for ``name`` bound to ``options``.

    ``notes``, if given, collects lossy-but-kept conversions (e.g.
    ``lossy_multiturn_flattened``); the caller clears it between examples.
    """
    if name not in EMITTERS:
        load_entry_points(EMITTER_GROUP, EMITTERS)
    if name not in EMITTERS:
        raise ValueError(unknown_name_message("output format", name, list(EMITTERS)))
    if tool_arguments is not None:
        options = replace(options or EmitOptions(), tool_arguments=tool_arguments)
    fn = EMITTERS[name]
    if fn is emit_messages:
        return partial(emit_messages, options=options) if options else emit_messages
    if fn is emit_alpaca:
        return partial(emit_alpaca, options=options, notes=notes)
    if fn is emit_preference:
        return partial(emit_preference, options=options) if options else emit_preference
    if fn is emit_sharegpt:
        return partial(emit_sharegpt, options=options, notes=notes)
    if fn is emit_sharegpt_preference:
        return partial(emit_sharegpt_preference, options=options) if options else fn
    # Plugin formats get the options only if they declare an ``options`` parameter.
    if options is not None and "options" in inspect.signature(fn).parameters:
        return partial(fn, options=options)  # type: ignore[call-arg]
    return fn


def register_emitter(name: str, fn: EmitterFn, *, replace: bool = False) -> None:
    """Make ``fn`` available as ``--format name``.

    ``fn`` takes a :class:`TrainingExample` and returns the JSON object for one
    output line; it may accept an ``options: EmitOptions`` keyword and may raise
    :class:`UnrepresentableExample` to have an example dropped and counted.
    Registering an existing name raises unless ``replace=True``.
    """
    if not replace and name in EMITTERS:
        raise ValueError(f"output format {name!r} is already registered (pass replace=True)")
    EMITTERS[name] = fn


def available_formats() -> list[str]:
    """Built-in, registered, and entry-point output format names."""
    load_entry_points(EMITTER_GROUP, EMITTERS)
    return sorted(EMITTERS)
