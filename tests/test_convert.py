"""Conversion pipeline and CLI integration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from convmerge.adapter_resolve import resolve_adapter
from convmerge.config import AdapterOptions, ChatAdapterOptions
from convmerge.convert import convert_file
from convmerge.emitters import get_emitter


def _iter_converted_lines(
    lines: Iterator[str],
    *,
    adapter_name: str,
    output_format: str,
    adapter_options: AdapterOptions | None = None,
) -> Iterator[str]:
    """Adapter then emitter on in-memory lines, without validation."""
    adapter = resolve_adapter(adapter_name, adapter_options)
    emitter = get_emitter(output_format)
    for raw in lines:
        obj = json.loads(raw)
        for example in adapter(obj):
            yield json.dumps(emitter(example), ensure_ascii=False)


def test_iter_chat_pairwise_both() -> None:
    line = json.dumps(
        {
            "conversation_a": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ],
            "conversation_b": [
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
            ],
            "winner": "model_a",
        }
    )
    opts = AdapterOptions(chat=ChatAdapterOptions(pairwise_mode="both"))
    out = list(
        _iter_converted_lines(
            iter([line]),
            adapter_name="chat",
            output_format="messages",
            adapter_options=opts,
        )
    )
    assert len(out) == 2


def test_iter_alpaca_to_messages() -> None:
    lines = ['{"instruction": "Hi", "input": "", "output": "Hey"}']
    out = list(
        _iter_converted_lines(
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


def test_convert_file_stats_counts_skip_reasons(tmp_path: Path) -> None:
    from convmerge.convert import ConvertStats

    src = tmp_path / "in.jsonl"
    src.write_text(
        '{"instruction": "a", "output": "b"}\n\n{bad json\n[1, 2]\n{"unrelated": 1}\n',
        encoding="utf-8",
    )
    stats = ConvertStats()
    n_in, n_out = convert_file(
        src,
        tmp_path / "out.jsonl",
        adapter_name="alpaca",
        output_format="messages",
        stats=stats,
    )
    assert (n_in, n_out) == (5, 1)
    assert stats.blank == 1
    assert stats.invalid_json == 1
    assert stats.first_invalid_line == 3
    assert stats.non_object == 1
    assert stats.no_example == 1
    assert stats.skipped == 3


def test_convert_with_config_passes_progress(tmp_path: Path, capsys) -> None:
    from convmerge.config import ConvertConfig
    from convmerge.convert import convert_with_config

    src = tmp_path / "in.jsonl"
    src.write_text('{"instruction": "a", "output": "b"}\n', encoding="utf-8")
    convert_with_config(
        src, tmp_path / "out.jsonl", ConvertConfig("alpaca", "messages"), progress=True
    )
    assert "[done] convert in.jsonl" in capsys.readouterr().err
