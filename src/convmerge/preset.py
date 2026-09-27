"""YAML/JSON convert presets and template generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from convmerge.adapters.mapped import MapSpec
from convmerge.config import (
    AdapterOptions,
    ConvertConfig,
    chat_adapter_options_from_mapping,
    check_preference,
    emit_options_from_mapping,
    sharegpt_adapter_options_from_mapping,
    transform_options_from_mapping,
)

PRESET_TEMPLATE_YAML = """# convmerge convert preset (v1)
# See docs/custom_presets.md in the convmerge repository.

adapter: chat
output_format: messages
encoding: utf-8

# Optional: tune the auto/chat adapter (see convmerge.adapters.chat.iter_from_chat_line)
adapter_options:
  chat:
    # pairwise_mode: winner   # winner | both | a | b
    # conversation_keys: [messages, conversation, conversations]
    # role_keys: [role, from]
    # content_keys: [content, value, text]
    # role_map:
    #   human: user
    #   gpt: assistant
  # preference: chosen        # DPO / reward data: train on the chosen answer
  # Optional: field mapping for adapter: map (dotted paths into each record)
  # map:
  #   user: question.text
  #   assistant: answer.text
  # Optional: sharegpt adapter (adapter: sharegpt)
  # sharegpt:
  #   turn_mode: full         # full (default, whole conversation) | pairs (0.5.x behavior)

# Optional: output format options
# output_options:
#   tool_arguments: string    # string (OpenAI) | object
#   keep_meta: false          # true, or a list of keys such as [source, id]
#   meta_key: meta
#   alpaca_multiturn: flatten # flatten | history | drop
#   reasoning: keep           # keep | inline | reasoning_content | thinking | drop
#   tool_content: empty       # empty ("" on tool-call turns) | null

# Optional: fixes for strict chat templates (all off by default)
# transforms:
#   system: keep              # keep | fold (into the first user turn) | drop
#   merge_consecutive: false  # join consecutive same-role turns
#   split_turns: false        # one example per user turn
#   reasoning_turns: all      # all | last (reasoning only after the last user turn)
"""


def _require_yaml() -> Any:
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "Preset files require PyYAML. Install with: pip install 'convmerge[preset]' (or [all]) "
            "or pip install pyyaml"
        ) from e
    return yaml


def load_raw_preset(path: Path) -> dict[str, Any]:
    """Load preset file as a dict (YAML or JSON)."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".json",):
        data = json.loads(text)
    else:
        yaml = _require_yaml()
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"preset root must be a mapping, got {type(data).__name__}")
    return data


def load_convert_preset(path: Path) -> ConvertConfig:
    """Load a preset file into :class:`convmerge.config.ConvertConfig`."""
    data = load_raw_preset(path)
    adapter = data.get("adapter")
    output_format = data.get("output_format")
    encoding = data.get("encoding") or "utf-8"
    adapter_options: AdapterOptions | None = None
    ao = data.get("adapter_options")
    if ao is not None:
        if not isinstance(ao, dict):
            raise ValueError("adapter_options must be a mapping")
        ch = ao.get("chat")
        sg = ao.get("sharegpt")
        if ch is not None and not isinstance(ch, dict):
            raise ValueError("adapter_options.chat must be a mapping")
        if sg is not None and not isinstance(sg, dict):
            raise ValueError("adapter_options.sharegpt must be a mapping")
        pref = ao.get("preference")
        if pref is not None:
            pref = check_preference(pref)
        mp = ao.get("map")
        spec = MapSpec.from_mapping(mp) if mp is not None else None
        if ch is not None or sg is not None or pref is not None or spec is not None:
            adapter_options = AdapterOptions(
                chat=chat_adapter_options_from_mapping(ch) if ch is not None else None,
                sharegpt=sharegpt_adapter_options_from_mapping(sg) if sg is not None else None,
                preference=pref,
                map=spec,
            )
    emit_options = None
    oo = data.get("output_options")
    if oo is not None:
        if not isinstance(oo, dict):
            raise ValueError("output_options must be a mapping")
        emit_options = emit_options_from_mapping(oo)
    transform_options = None
    tr = data.get("transforms")
    if tr is not None:
        if not isinstance(tr, dict):
            raise ValueError("transforms must be a mapping")
        transform_options = transform_options_from_mapping(tr)
    if not isinstance(adapter, str) or not adapter.strip():
        raise ValueError("preset requires non-empty string 'adapter'")
    if not isinstance(output_format, str) or not output_format.strip():
        raise ValueError("preset requires non-empty string 'output_format'")
    if not isinstance(encoding, str) or not encoding.strip():
        raise ValueError("preset 'encoding' must be a non-empty string when set")
    return ConvertConfig(
        adapter=adapter.strip(),
        output_format=output_format.strip(),
        encoding=encoding.strip(),
        adapter_options=adapter_options,
        emit_options=emit_options,
        transform_options=transform_options,
    )


def validate_preset_file(path: Path) -> None:
    """Raise ValueError with a clear message if the preset is invalid."""
    from convmerge.adapters import available_adapters
    from convmerge.emitters import available_formats

    cfg = load_convert_preset(path)
    adapters = available_adapters()
    formats = available_formats()
    if cfg.adapter not in adapters:
        known = ", ".join(adapters)
        raise ValueError(f"unknown adapter {cfg.adapter!r}. Choose one of: {known}")
    if cfg.output_format not in formats:
        known = ", ".join(formats)
        raise ValueError(f"unknown output_format {cfg.output_format!r}. Choose one of: {known}")
    if cfg.adapter_options and cfg.adapter_options.chat:
        pm = cfg.adapter_options.chat.pairwise_mode
        if pm not in ("winner", "both", "a", "b"):
            raise ValueError(
                f"invalid pairwise_mode {pm!r}; use winner, both, a, or b",
            )
