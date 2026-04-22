"""Preset loading, validation, and convert config merge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.config import (
    AdapterOptions,
    ChatAdapterOptions,
    build_convert_config,
)
from convmerge.convert import convert_with_config
from convmerge.preset import load_convert_preset, validate_preset_file


def test_load_convert_preset_yaml(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text(
        """
adapter: chat
output_format: messages
encoding: utf-8
adapter_options:
  chat:
    pairwise_mode: both
""",
        encoding="utf-8",
    )
    cfg = load_convert_preset(p)
    assert cfg.adapter == "chat"
    assert cfg.output_format == "messages"
    assert cfg.encoding == "utf-8"
    assert cfg.adapter_options is not None
    assert cfg.adapter_options.chat is not None
    assert cfg.adapter_options.chat.pairwise_mode == "both"


def test_build_convert_config_preset_overridden_by_flags(tmp_path: Path) -> None:
    preset = tmp_path / "p.yaml"
    preset.write_text(
        "adapter: chat\noutput_format: messages\n",
        encoding="utf-8",
    )
    cfg = build_convert_config(
        preset_path=preset,
        adapter="alpaca",
        output_format="alpaca",
    )
    assert cfg.adapter == "alpaca"
    assert cfg.output_format == "alpaca"


def test_build_convert_config_adapter_kwargs_merge(tmp_path: Path) -> None:
    preset = tmp_path / "p.yaml"
    preset.write_text(
        """
adapter: chat
output_format: messages
adapter_options:
  chat:
    pairwise_mode: winner
""",
        encoding="utf-8",
    )
    cfg = build_convert_config(
        preset_path=preset,
        adapter_kwargs_json=json.dumps({"chat": {"pairwise_mode": "both"}}),
    )
    assert cfg.adapter_options is not None
    assert cfg.adapter_options.chat is not None
    assert cfg.adapter_options.chat.pairwise_mode == "both"


def test_validate_preset_file_ok(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text("adapter: chat\noutput_format: messages\n", encoding="utf-8")
    validate_preset_file(p)


def test_validate_preset_file_bad_adapter(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text("adapter: not_real\noutput_format: messages\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown adapter"):
        validate_preset_file(p)


def test_convert_with_config_roundtrip(tmp_path: Path) -> None:
    from convmerge.preset import load_convert_preset

    src = Path(__file__).parent / "fixtures" / "alpaca_one.jsonl"
    preset = tmp_path / "p.yaml"
    preset.write_text("adapter: alpaca\noutput_format: messages\n", encoding="utf-8")
    cfg = load_convert_preset(preset)
    dst = tmp_path / "out.jsonl"
    n_in, n_out = convert_with_config(src, dst, cfg)
    assert n_in >= 1
    assert n_out >= 1
    assert "messages" in json.loads(dst.read_text(encoding="utf-8").strip())


def test_api_adapter_options_overlay(tmp_path: Path) -> None:
    preset = tmp_path / "p.yaml"
    preset.write_text(
        """
adapter: chat
output_format: messages
adapter_options:
  chat:
    pairwise_mode: winner
""",
        encoding="utf-8",
    )
    cfg = build_convert_config(
        preset_path=preset,
        adapter_options=AdapterOptions(chat=ChatAdapterOptions(pairwise_mode="both")),
    )
    assert cfg.adapter_options is not None
    assert cfg.adapter_options.chat is not None
    assert cfg.adapter_options.chat.pairwise_mode == "both"
