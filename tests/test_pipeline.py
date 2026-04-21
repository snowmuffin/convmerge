"""Integration tests for convmerge.pipeline.build_sft_jsonl and convert_dir."""

from __future__ import annotations

import json
from pathlib import Path

from convmerge.convert import convert_dir
from convmerge.pipeline import build_sft_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_convert_dir_preserves_tree(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    _write_jsonl(
        src / "sub" / "a.jsonl",
        [{"instruction": "q1", "output": "a1"}, {"instruction": "q2", "output": "a2"}],
    )
    dst = tmp_path / "dst"

    n_files, n_in, n_out = convert_dir(src, dst, adapter_name="alpaca", output_format="messages")
    assert n_files == 1
    assert n_in == 2
    assert n_out == 2

    out = (dst / "sub" / "a.jsonl").read_text().splitlines()
    assert len(out) == 2
    first = json.loads(out[0])
    assert "messages" in first
    assert first["messages"][0]["role"] == "user"


def test_build_sft_jsonl_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    # Alpaca-style JSON array
    (raw / "alpaca.json").write_text(
        json.dumps([{"instruction": f"q{i}", "output": f"a{i}"} for i in range(20)])
    )
    # ShareGPT-style JSONL
    sharegpt_rows = [
        {
            "messages": [
                {"role": "user", "content": f"hello {i}"},
                {"role": "assistant", "content": f"hi {i}"},
            ]
        }
        for i in range(20)
    ]
    _write_jsonl(raw / "chat.jsonl", sharegpt_rows)

    out = tmp_path / "out"
    result = build_sft_jsonl(raw, out, train_ratio=0.8, seed=123, min_turns=1)

    assert result.counts["normalize_files"] == 2
    assert result.counts["normalize_rows"] == 40
    assert result.train_path.exists()
    assert result.test_path.exists()
    # Total = train + test <= 40 (after dedupe / filter).
    train_lines = result.train_path.read_text().strip().splitlines()
    test_lines = result.test_path.read_text().strip().splitlines()
    assert len(train_lines) + len(test_lines) == result.counts["filter_kept"]
    # Every output row should be messages-shaped.
    for ln in train_lines + test_lines:
        obj = json.loads(ln)
        assert "messages" in obj
        assert isinstance(obj["messages"], list)
