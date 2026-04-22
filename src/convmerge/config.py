"""Convert pipeline configuration (adapter options, presets)."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from convmerge.adapters.chat import (
    DEFAULT_CONTENT_KEYS,
    DEFAULT_CONVERSATION_KEYS,
    DEFAULT_ROLE_KEYS,
)


@dataclass
class ChatAdapterOptions:
    """Options passed to :func:`convmerge.adapters.chat.iter_from_chat_line`."""

    conversation_keys: tuple[str, ...] = DEFAULT_CONVERSATION_KEYS
    role_keys: tuple[str, ...] = DEFAULT_ROLE_KEYS
    content_keys: tuple[str, ...] = DEFAULT_CONTENT_KEYS
    role_map: dict[str, str] | None = None
    pairwise_mode: str = "winner"
    instruction_keys: tuple[str, ...] = ("instruction", "question", "prompt")
    output_keys: tuple[str, ...] = ("output", "response", "answer")
    input_keys: tuple[str, ...] = ("input", "context")


@dataclass
class AdapterOptions:
    """Per-adapter tuning; only ``chat`` is supported in v1."""

    chat: ChatAdapterOptions | None = None


@dataclass
class ConvertConfig:
    """Resolved settings for :func:`convmerge.convert.convert_file`."""

    adapter: str
    output_format: str
    encoding: str = "utf-8"
    adapter_options: AdapterOptions | None = None


def _as_tuple_str(v: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(v, tuple):
        return tuple(str(x) for x in v)
    if isinstance(v, list):
        return tuple(str(x) for x in v)
    raise ValueError(f"{field_name}: expected a list of strings, got {type(v).__name__}")


def chat_adapter_options_from_mapping(data: dict[str, Any]) -> ChatAdapterOptions:
    """Build :class:`ChatAdapterOptions` from a YAML/JSON mapping (partial ok)."""
    kw: dict[str, Any] = {}
    if "conversation_keys" in data:
        kw["conversation_keys"] = _as_tuple_str(
            data["conversation_keys"],
            field_name="conversation_keys",
        )
    if "role_keys" in data:
        kw["role_keys"] = _as_tuple_str(data["role_keys"], field_name="role_keys")
    if "content_keys" in data:
        kw["content_keys"] = _as_tuple_str(data["content_keys"], field_name="content_keys")
    if "instruction_keys" in data:
        kw["instruction_keys"] = _as_tuple_str(
            data["instruction_keys"],
            field_name="instruction_keys",
        )
    if "output_keys" in data:
        kw["output_keys"] = _as_tuple_str(data["output_keys"], field_name="output_keys")
    if "input_keys" in data:
        kw["input_keys"] = _as_tuple_str(data["input_keys"], field_name="input_keys")
    if "pairwise_mode" in data:
        kw["pairwise_mode"] = str(data["pairwise_mode"])
    if "role_map" in data:
        rm = data["role_map"]
        if rm is not None and not isinstance(rm, dict):
            raise ValueError("role_map must be a string->string mapping or null")
        kw["role_map"] = None if rm is None else {str(k): str(v) for k, v in rm.items()}
    base = ChatAdapterOptions()
    return replace(base, **kw)


def _chat_options_to_override_dict(chat: ChatAdapterOptions) -> dict[str, Any]:
    """Fields that differ from defaults become a merge dict."""
    defaults = ChatAdapterOptions()
    out: dict[str, Any] = {}
    for f in fields(ChatAdapterOptions):
        v = getattr(chat, f.name)
        d = getattr(defaults, f.name)
        if v != d:
            if isinstance(v, tuple):
                out[f.name] = list(v)
            else:
                out[f.name] = v
    return out


def _merge_chat_dicts(*layers: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(layer)
    return merged


def build_convert_config(
    *,
    preset_path: Path | None = None,
    adapter: str | None = None,
    output_format: str | None = None,
    encoding: str | None = None,
    adapter_options: AdapterOptions | None = None,
    adapter_kwargs_json: str | None = None,
) -> ConvertConfig:
    """
    Merge preset file, explicit CLI/API arguments, and optional JSON adapter kwargs.

    Order for ``adapter_options.chat`` fields: preset, then ``--adapter-kwargs``,
    then explicit ``adapter_options``.
    Explicit ``adapter`` / ``output_format`` / ``encoding`` override the preset.
    """
    from convmerge.preset import load_convert_preset

    cfg_adapter: str | None = None
    cfg_format: str | None = None
    cfg_encoding: str | None = None
    chat_layers: list[dict[str, Any]] = []

    if preset_path is not None:
        p = load_convert_preset(preset_path)
        cfg_adapter = p.adapter
        cfg_format = p.output_format
        cfg_encoding = p.encoding
        if p.adapter_options and p.adapter_options.chat:
            chat_layers.append(_chat_options_to_override_dict(p.adapter_options.chat))

    if adapter_kwargs_json:
        try:
            raw = json.loads(adapter_kwargs_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid --adapter-kwargs JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError("--adapter-kwargs must be a JSON object")
        ch = raw.get("chat")
        if ch is not None:
            if not isinstance(ch, dict):
                raise ValueError("--adapter-kwargs: 'chat' must be an object")
            chat_layers.append(ch)

    if adapter_options and adapter_options.chat:
        chat_layers.append(_chat_options_to_override_dict(adapter_options.chat))

    if adapter is not None:
        cfg_adapter = adapter
    if output_format is not None:
        cfg_format = output_format
    if encoding is not None:
        cfg_encoding = encoding

    if not cfg_adapter or not cfg_format:
        raise ValueError(
            "adapter and output format are required (via --preset or --from / --format)."
        )

    adapter_opts: AdapterOptions | None = None
    if chat_layers:
        merged_chat = chat_adapter_options_from_mapping(_merge_chat_dicts(*chat_layers))
        adapter_opts = AdapterOptions(chat=merged_chat)

    return ConvertConfig(
        adapter=cfg_adapter,
        output_format=cfg_format,
        encoding=cfg_encoding or "utf-8",
        adapter_options=adapter_opts,
    )
