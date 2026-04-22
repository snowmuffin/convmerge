"""CLI: preset subcommands and convert --preset."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cli_preset_init_stdout() -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    r = subprocess.run(
        [sys.executable, "-m", "convmerge", "preset", "init"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "adapter:" in r.stdout
    assert "output_format:" in r.stdout


def test_cli_convert_with_preset(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    src = Path(__file__).parent / "fixtures" / "alpaca_one.jsonl"
    dst = tmp_path / "out.jsonl"
    preset = tmp_path / "p.yaml"
    preset.write_text("adapter: alpaca\noutput_format: messages\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
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
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert dst.read_text(encoding="utf-8").strip()


def test_cli_preset_validate_ok(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    preset = tmp_path / "p.yaml"
    preset.write_text("adapter: chat\noutput_format: messages\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    r = subprocess.run(
        [sys.executable, "-m", "convmerge", "preset", "validate", str(preset)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "ok" in r.stderr
