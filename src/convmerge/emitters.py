"""TrainingExample → target JSON object for JSONL lines."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal

from convmerge.models import ChatMessage, ContentPart, ToolCall, TrainingExample
from convmerge.plugins import EMITTER_GROUP, load_entry_points

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
    """

    tool_arguments: ToolArguments = "string"
    keep_meta: bool | Sequence[str] = False
    meta_key: str = "meta"
    alpaca_multiturn: AlpacaMultiturn = "flatten"

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
        if not isinstance(self.keep_meta, bool):
            object.__setattr__(self, "keep_meta", tuple(self.keep_meta))


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
    if options is not None:
        tool_arguments = options.tool_arguments
    row: dict[str, Any] = {
        "messages": [_message_dict(m, tool_arguments) for m in example.messages],
    }
    if example.tools:
        row["tools"] = example.tools
    return _with_meta(row, example, options)


def _with_meta(
    row: dict[str, Any], example: TrainingExample, options: EmitOptions | None
) -> dict[str, Any]:
    if options is None or options.keep_meta is False:
        return row
    meta = dict(example.meta)
    if options.keep_meta is not True:
        meta = {k: v for k, v in meta.items() if k in options.keep_meta}
    if meta:
        row[options.meta_key] = meta
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
    msgs = example.messages
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
    args = opts.tool_arguments
    row: dict[str, Any] = {
        "prompt": [_message_dict(m, args) for m in prompt],
        "chosen": [_message_dict(m, args) for m in chosen_tail],
        "rejected": [_message_dict(m, args) for m in rejected_tail],
    }
    if example.tools:
        row["tools"] = example.tools
    return _with_meta(row, example, options)


# Asks the adapter to keep both answers of preference records (see
# :func:`convmerge.adapters.preference.iter_pairs`).
emit_preference.preference_pairs = True  # type: ignore[attr-defined]


def _has_answer(turns: list[ChatMessage]) -> bool:
    return any(m.role == "assistant" and (m.text.strip() or m.tool_calls) for m in turns)


def wants_pairs(output_format: str) -> bool:
    """Whether ``output_format`` consumes chosen/rejected pairs."""
    fn = EMITTERS.get(output_format)
    return bool(getattr(fn, "preference_pairs", False))


EMITTERS: dict[str, EmitterFn] = {
    "messages": emit_messages,
    "alpaca": emit_alpaca,
    "preference": emit_preference,
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
        known = ", ".join(sorted(EMITTERS))
        raise ValueError(f"Unknown output format {name!r}. Choose one of: {known}")
    if tool_arguments is not None:
        options = replace(options or EmitOptions(), tool_arguments=tool_arguments)
    fn = EMITTERS[name]
    if fn is emit_messages:
        return partial(emit_messages, options=options) if options else emit_messages
    if fn is emit_alpaca:
        return partial(emit_alpaca, options=options, notes=notes)
    if fn is emit_preference:
        return partial(emit_preference, options=options) if options else emit_preference
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
