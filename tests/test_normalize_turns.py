"""Tests for convmerge.normalize.turns."""

from __future__ import annotations

import json
from pathlib import Path

from convmerge.normalize.turns import (
    analyze_turn_distribution,
    count_turns,
    is_single_turn,
    split_by_turns,
)


def _make_sample(turns: int) -> dict:
    messages = []
    for _ in range(turns):
        messages.append({"role": "user", "content": "q"})
        messages.append({"role": "assistant", "content": "a"})
    return {"messages": messages}


def test_count_turns_and_single() -> None:
    assert count_turns(_make_sample(1)) == 1
    assert count_turns(_make_sample(3)) == 3
    assert is_single_turn(_make_sample(1)) is True
    assert is_single_turn(_make_sample(2)) is False


def test_analyze_turn_distribution(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for t in (1, 1, 2, 3, 3, 3):
            f.write(json.dumps(_make_sample(t)) + "\n")
    report = analyze_turn_distribution(p)
    assert report["total"] == 6
    assert report["single"] == 2
    assert report["multi"] == 4
    assert report["distribution"] == {1: 2, 2: 1, 3: 3}


def test_split_by_turns(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    with src.open("w", encoding="utf-8") as f:
        for t in (1, 2, 1, 3):
            f.write(json.dumps(_make_sample(t)) + "\n")

    single_out = tmp_path / "single.jsonl"
    multi_out = tmp_path / "multi.jsonl"
    s, m = split_by_turns(src, single_out=single_out, multi_out=multi_out)
    assert (s, m) == (2, 2)
    assert sum(1 for _ in single_out.open(encoding="utf-8")) == 2
    assert sum(1 for _ in multi_out.open(encoding="utf-8")) == 2
