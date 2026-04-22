"""Integration tests: directory normalize + convert with YAML preset (typical SFT prep)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_cli_normalize_mirrors_directory_tree(tmp_path: Path) -> None:
    """``convmerge normalize -i dir -o dir`` preserves relative paths under the output root."""
    raw = tmp_path / "raw"
    (raw / "nested").mkdir(parents=True)
    (raw / "nested" / "sample.json").write_text('[{"k": 1}, {"k": 2}]', encoding="utf-8")
    out = tmp_path / "jsonl"
    env = {**os.environ, "PYTHONPATH": str(_root() / "src")}
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "convmerge",
            "normalize",
            "-i",
            str(raw),
            "-o",
            str(out),
        ],
        cwd=_root(),
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    dst = out / "nested" / "sample.jsonl"
    assert dst.is_file(), r.stderr
    lines = [ln for ln in dst.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"k": 1}


def test_cli_convert_preset_alpaca_to_messages(tmp_path: Path) -> None:
    """CLI ``convert --preset`` with alpaca -> messages matches explicit adapter flags."""
    src = Path(__file__).parent / "fixtures" / "alpaca_one.jsonl"
    preset = tmp_path / "preset.yaml"
    preset.write_text(
        "adapter: alpaca\noutput_format: messages\nencoding: utf-8\n",
        encoding="utf-8",
    )
    dst = tmp_path / "out.jsonl"
    env = {**os.environ, "PYTHONPATH": str(_root() / "src")}
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "convmerge",
            "convert",
            "-i",
            str(src),
            "-o",
            str(dst),
            "--preset",
            str(preset),
        ],
        cwd=_root(),
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    row = json.loads(dst.read_text(encoding="utf-8").strip())
    assert "messages" in row
    assert row["messages"][0]["role"] == "user"


def test_convert_with_config_matches_explicit_convert_file(tmp_path: Path) -> None:
    """``convert_with_config`` from a preset file matches ``convert_file`` with explicit args."""
    from convmerge.config import build_convert_config
    from convmerge.convert import convert_file, convert_with_config
    from convmerge.preset import load_convert_preset

    src = Path(__file__).parent / "fixtures" / "alpaca_one.jsonl"
    preset = tmp_path / "preset.yaml"
    preset.write_text("adapter: alpaca\noutput_format: messages\n", encoding="utf-8")
    cfg = load_convert_preset(preset)
    cfg2 = build_convert_config(preset_path=preset)
    assert cfg.adapter == cfg2.adapter
    assert cfg.output_format == cfg2.output_format

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    convert_file(
        src,
        a,
        adapter_name="alpaca",
        output_format="messages",
    )
    convert_with_config(src, b, cfg)
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
