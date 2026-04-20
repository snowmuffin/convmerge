"""Conversion pipeline and CLI integration."""

from __future__ import annotations

import json
from pathlib import Path

from convmerge.convert import convert_file, iter_converted_lines


def test_iter_alpaca_to_messages() -> None:
    lines = ['{"instruction": "Hi", "input": "", "output": "Hey"}']
    out = list(
        iter_converted_lines(
            iter(lines),
            adapter_name="alpaca",
            output_format="messages",
        )
    )
    assert len(out) == 1
    obj = json.loads(out[0])
    assert "messages" in obj
    assert obj["messages"][0]["role"] == "user"
    assert obj["messages"][1]["role"] == "assistant"


def test_convert_file_roundtrip(tmp_path: Path) -> None:
    src = Path(__file__).parent / "fixtures" / "alpaca_one.jsonl"
    dst = tmp_path / "out.jsonl"
    n_in, n_out = convert_file(
        src,
        dst,
        adapter_name="alpaca",
        output_format="messages",
    )
    assert n_in >= 1
    assert n_out >= 1
    text = dst.read_text(encoding="utf-8").strip()
    obj = json.loads(text)
    assert "messages" in obj


def test_cli_convert(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    src = Path(__file__).parent / "fixtures" / "alpaca_one.jsonl"
    dst = tmp_path / "cli_out.jsonl"
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
            "--from",
            "alpaca",
            "--format",
            "messages",
        ],
        check=False,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert dst.read_text(encoding="utf-8").strip()
    assert "wrote" in r.stderr
