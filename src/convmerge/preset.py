"""YAML/JSON convert presets and template generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from convmerge.config import AdapterOptions, ConvertConfig, chat_adapter_options_from_mapping

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
"""


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "Preset files require PyYAML. Install with: pip install 'convmerge[preset]' "
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
        if ch is not None:
            if not isinstance(ch, dict):
                raise ValueError("adapter_options.chat must be a mapping")
            adapter_options = AdapterOptions(chat=chat_adapter_options_from_mapping(ch))
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
    )


def validate_preset_file(path: Path) -> None:
    """Raise ValueError with a clear message if the preset is invalid."""
    from convmerge.adapters import ADAPTERS
    from convmerge.emitters import EMITTERS

    cfg = load_convert_preset(path)
    if cfg.adapter not in ADAPTERS:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"unknown adapter {cfg.adapter!r}. Choose one of: {known}")
    if cfg.output_format not in EMITTERS:
        known = ", ".join(sorted(EMITTERS))
        raise ValueError(f"unknown output_format {cfg.output_format!r}. Choose one of: {known}")
    if cfg.adapter_options and cfg.adapter_options.chat:
        pm = cfg.adapter_options.chat.pairwise_mode
        if pm not in ("winner", "both", "a", "b"):
            raise ValueError(
                f"invalid pairwise_mode {pm!r}; use winner, both, a, or b",
            )
