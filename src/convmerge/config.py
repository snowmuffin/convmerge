"""Convert pipeline configuration (adapter options, presets)."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from convmerge.adapters._common import DEFAULT_REASONING_KEYS
from convmerge.adapters.chat import (
    DEFAULT_CONTENT_KEYS,
    DEFAULT_CONVERSATION_KEYS,
    DEFAULT_INPUT_KEYS,
    DEFAULT_INSTRUCTION_KEYS,
    DEFAULT_OUTPUT_KEYS,
    DEFAULT_ROLE_KEYS,
)
from convmerge.adapters.mapped import MapSpec
from convmerge.emitters import EmitOptions
from convmerge.transforms import TransformOptions


@dataclass
class ChatAdapterOptions:
    """Options passed to :func:`convmerge.adapters.chat.iter_from_chat_line`."""

    conversation_keys: tuple[str, ...] = DEFAULT_CONVERSATION_KEYS
    role_keys: tuple[str, ...] = DEFAULT_ROLE_KEYS
    content_keys: tuple[str, ...] = DEFAULT_CONTENT_KEYS
    role_map: dict[str, str] | None = None
    pairwise_mode: str = "winner"
    instruction_keys: tuple[str, ...] = DEFAULT_INSTRUCTION_KEYS
    output_keys: tuple[str, ...] = DEFAULT_OUTPUT_KEYS
    input_keys: tuple[str, ...] = DEFAULT_INPUT_KEYS
    reasoning_keys: tuple[str, ...] = DEFAULT_REASONING_KEYS
    record_reasoning_keys: tuple[str, ...] = ()


@dataclass
class SharegptAdapterOptions:
    """Options passed to :func:`convmerge.adapters.sharegpt.iter_from_sharegpt_line`.

    ``turn_mode=None`` means the adapter default (``"full"`` since 0.6.0);
    ``"pairs"`` restores the 0.5.x one-example-per-pair behavior.
    """

    turn_mode: str | None = None


@dataclass
class AdapterOptions:
    """Per-adapter tuning for the ``chat``/``auto`` and ``sharegpt`` adapters."""

    chat: ChatAdapterOptions | None = None
    sharegpt: SharegptAdapterOptions | None = None
    preference: str | None = None
    """``"chosen"`` / ``"rejected"``: fold that answer of a preference record
    into the conversation before adapting (see :mod:`convmerge.adapters.preference`)."""
    map: MapSpec | None = None
    """Field mapping for ``--from map`` (see :mod:`convmerge.adapters.mapped`)."""


@dataclass
class ConvertConfig:
    """Resolved settings for :func:`convmerge.convert.convert_file`."""

    adapter: str
    output_format: str
    encoding: str = "utf-8"
    adapter_options: AdapterOptions | None = None
    emit_options: EmitOptions | None = None
    transform_options: TransformOptions | None = None


_EMIT_OPTION_KEYS = (
    "tool_arguments",
    "keep_meta",
    "meta_key",
    "alpaca_multiturn",
    "reasoning",
    "tool_content",
    "meta",
)
_TRANSFORM_OPTION_KEYS = (
    "system",
    "merge_consecutive",
    "split_turns",
    "reasoning_turns",
    "leading_assistant",
)


def check_preference(value: Any) -> str:
    from convmerge.adapters.preference import PREFERENCES

    if value not in PREFERENCES:
        raise ValueError(f"preference must be one of {list(PREFERENCES)}, got {value!r}")
    return value


def emit_options_from_mapping(data: dict[str, Any]) -> EmitOptions:
    """Build :class:`EmitOptions` from a preset's ``output_options`` mapping."""
    unknown = set(data) - set(_EMIT_OPTION_KEYS)
    if unknown:
        raise ValueError(
            f"output_options: unknown option(s) {sorted(unknown)}; "
            f"supported: {', '.join(_EMIT_OPTION_KEYS)}"
        )
    kw: dict[str, Any] = {}
    for key in ("tool_arguments", "meta_key", "alpaca_multiturn", "reasoning", "tool_content"):
        if key in data:
            kw[key] = str(data[key])
    if "meta" in data:
        kw["meta_values"] = meta_values_from_mapping(data["meta"], where="output_options.meta")
    if "keep_meta" in data:
        km = data["keep_meta"]
        if isinstance(km, bool):
            kw["keep_meta"] = km
        elif isinstance(km, list):
            kw["keep_meta"] = tuple(str(k) for k in km)
        else:
            raise ValueError("output_options.keep_meta must be true/false or a list of keys")
    return EmitOptions(**kw)


def meta_values_from_mapping(data: Any, *, where: str = "meta") -> dict[str, str]:
    """Constant ``meta`` fields: a mapping of names to scalar values (kept as text)."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping of names to values")
    out: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(v, (dict, list)) or v is None:
            raise ValueError(f"{where}.{k}: expected a string or number")
        out[str(k)] = str(v)
    return out


def transform_options_from_mapping(
    data: dict[str, Any], *, where: str = "transforms"
) -> TransformOptions:
    """Build :class:`TransformOptions` from a preset's or recipe's mapping."""
    unknown = set(data) - set(_TRANSFORM_OPTION_KEYS)
    if unknown:
        raise ValueError(
            f"{where}: unknown option(s) {sorted(unknown)}; "
            f"supported: {', '.join(_TRANSFORM_OPTION_KEYS)}"
        )
    kw: dict[str, Any] = {}
    for key in ("merge_consecutive", "split_turns"):
        if key in data:
            if not isinstance(data[key], bool):
                raise ValueError(f"{where}.{key} must be true or false")
            kw[key] = data[key]
    for key in ("system", "reasoning_turns", "leading_assistant"):
        if key in data:
            kw[key] = str(data[key])
    return TransformOptions(**kw)


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
    for key in ("reasoning_keys", "record_reasoning_keys"):
        if key in data:
            kw[key] = _as_tuple_str(data[key], field_name=key)
    if "pairwise_mode" in data:
        kw["pairwise_mode"] = str(data["pairwise_mode"])
    if "role_map" in data:
        rm = data["role_map"]
        if rm is not None and not isinstance(rm, dict):
            raise ValueError("role_map must be a string->string mapping or null")
        kw["role_map"] = None if rm is None else {str(k): str(v) for k, v in rm.items()}
    base = ChatAdapterOptions()
    return replace(base, **kw)


def sharegpt_adapter_options_from_mapping(data: dict[str, Any]) -> SharegptAdapterOptions:
    """Build :class:`SharegptAdapterOptions` from a YAML/JSON mapping."""
    from convmerge.adapters.sharegpt import TURN_MODES

    unknown = set(data) - {"turn_mode"}
    if unknown:
        raise ValueError(f"sharegpt: unknown option(s) {sorted(unknown)}; supported: turn_mode")
    mode = data.get("turn_mode")
    if mode is not None and mode not in TURN_MODES:
        raise ValueError(f"sharegpt.turn_mode must be one of {list(TURN_MODES)}, got {mode!r}")
    return SharegptAdapterOptions(turn_mode=mode)


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


def _sharegpt_options_to_override_dict(opts: SharegptAdapterOptions) -> dict[str, Any]:
    return {} if opts.turn_mode is None else {"turn_mode": opts.turn_mode}


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
    emit_overrides: dict[str, Any] | None = None,
    preference: str | None = None,
    transform_overrides: dict[str, Any] | None = None,
) -> ConvertConfig:
    """
    Merge preset file, explicit CLI/API arguments, and optional JSON adapter kwargs.

    Order for ``adapter_options.chat`` / ``adapter_options.sharegpt`` fields:
    preset, then ``--adapter-kwargs``, then explicit ``adapter_options``.
    Explicit ``adapter`` / ``output_format`` / ``encoding`` override the preset,
    and ``emit_overrides`` (keys of :class:`EmitOptions`) override the
    preset's ``output_options``. ``preference`` (``--preference``) overrides
    ``adapter_options.preference`` from the preset or ``--adapter-kwargs``.
    ``transform_overrides`` (keys of :class:`TransformOptions`) override the
    preset's ``transforms``. The output format defaults to ``messages``;
    an unknown adapter or format name raises ``ValueError``.
    """
    from convmerge.preset import load_convert_preset

    cfg_adapter: str | None = None
    cfg_format: str | None = None
    cfg_encoding: str | None = None
    cfg_emit: EmitOptions | None = None
    cfg_transforms: TransformOptions | None = None
    chat_layers: list[dict[str, Any]] = []
    sharegpt_layers: list[dict[str, Any]] = []
    cfg_preference: str | None = None
    cfg_map: MapSpec | None = None

    if preset_path is not None:
        p = load_convert_preset(preset_path)
        cfg_adapter = p.adapter
        cfg_format = p.output_format
        cfg_encoding = p.encoding
        cfg_emit = p.emit_options
        cfg_transforms = p.transform_options
        if p.adapter_options and p.adapter_options.chat:
            chat_layers.append(_chat_options_to_override_dict(p.adapter_options.chat))
        if p.adapter_options and p.adapter_options.sharegpt:
            sharegpt_layers.append(_sharegpt_options_to_override_dict(p.adapter_options.sharegpt))
        if p.adapter_options and p.adapter_options.preference:
            cfg_preference = p.adapter_options.preference
        if p.adapter_options and p.adapter_options.map:
            cfg_map = p.adapter_options.map

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
        sg = raw.get("sharegpt")
        if sg is not None:
            if not isinstance(sg, dict):
                raise ValueError("--adapter-kwargs: 'sharegpt' must be an object")
            sharegpt_layers.append(sg)
        if raw.get("preference") is not None:
            cfg_preference = check_preference(raw["preference"])
        if raw.get("map") is not None:
            cfg_map = MapSpec.from_mapping(raw["map"])

    if adapter_options and adapter_options.preference:
        cfg_preference = check_preference(adapter_options.preference)
    if preference is not None:
        cfg_preference = check_preference(preference)
    if adapter_options and adapter_options.chat:
        chat_layers.append(_chat_options_to_override_dict(adapter_options.chat))
    if adapter_options and adapter_options.sharegpt:
        sharegpt_layers.append(_sharegpt_options_to_override_dict(adapter_options.sharegpt))
    if adapter_options and adapter_options.map:
        cfg_map = adapter_options.map

    if adapter is not None:
        cfg_adapter = adapter
    if output_format is not None:
        cfg_format = output_format
    if encoding is not None:
        cfg_encoding = encoding

    if not cfg_adapter:
        raise ValueError(
            "a source adapter is required: --from auto (or alpaca, sharegpt, map, ...) "
            "or a preset that sets it"
        )
    cfg_format = cfg_format or "messages"
    _check_names(cfg_adapter, cfg_format)

    if cfg_adapter == "map" and cfg_map is None:
        raise ValueError(
            "--from map needs a field mapping: --adapter-kwargs "
            '\'{"map": {"user": "question", "assistant": "answer"}}\' (or map: in a recipe)'
        )

    adapter_opts: AdapterOptions | None = None
    if chat_layers or sharegpt_layers or cfg_preference or cfg_map:
        adapter_opts = AdapterOptions(
            preference=cfg_preference,
            map=cfg_map,
            chat=(
                chat_adapter_options_from_mapping(_merge_chat_dicts(*chat_layers))
                if chat_layers
                else None
            ),
            sharegpt=(
                sharegpt_adapter_options_from_mapping(_merge_chat_dicts(*sharegpt_layers))
                if sharegpt_layers
                else None
            ),
        )

    if cfg_preference:
        _check_preference_format(cfg_preference, cfg_format)

    if emit_overrides:
        cfg_emit = replace(cfg_emit or EmitOptions(), **emit_overrides)
    if transform_overrides:
        cfg_transforms = replace(cfg_transforms or TransformOptions(), **transform_overrides)
    if cfg_emit is not None:
        _check_reasoning_format(cfg_emit.reasoning, cfg_format)
    if cfg_transforms is not None:
        _check_split_turns(cfg_transforms, cfg_format)

    return ConvertConfig(
        adapter=cfg_adapter,
        output_format=cfg_format,
        encoding=cfg_encoding or "utf-8",
        adapter_options=adapter_opts,
        emit_options=cfg_emit,
        transform_options=cfg_transforms,
    )


# Built-in formats that have no place for a reasoning trace but inline text.
_INLINE_REASONING_FORMATS = frozenset({"alpaca", "sharegpt", "sharegpt-preference"})


def _check_names(adapter: str, output_format: str) -> None:
    """Fail early (a usage error) on an unknown adapter or format name."""
    from convmerge.adapters import get_adapter
    from convmerge.emitters import get_emitter

    get_adapter(adapter)
    get_emitter(output_format)


def _check_reasoning_format(reasoning: str, output_format: str) -> None:
    if reasoning in ("reasoning_content", "thinking") and output_format in (
        _INLINE_REASONING_FORMATS
    ):
        raise ValueError(
            f"reasoning={reasoning!r} (--reasoning) writes a separate field, but the "
            f"{output_format!r} format has none: use --reasoning inline (or keep / drop)"
        )


def _check_split_turns(transforms: TransformOptions, output_format: str) -> None:
    from convmerge.convert import check_transforms
    from convmerge.emitters import get_emitter, wants_pairs

    try:
        get_emitter(output_format)
    except ValueError:
        return
    check_transforms(transforms, pairs=wants_pairs(output_format))


def _check_preference_format(preference: str, output_format: str) -> None:
    from convmerge.emitters import get_emitter, wants_pairs

    try:
        get_emitter(output_format)  # loads plugin formats so wants_pairs sees them
    except ValueError:
        return  # unknown format: reported where the format is resolved
    if wants_pairs(output_format):
        raise ValueError(
            f"preference={preference!r} (--preference) turns pairs into SFT examples, but "
            f"the {output_format!r} format writes chosen/rejected pairs: use one or the other"
        )
