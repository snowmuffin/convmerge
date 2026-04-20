"""Adapter unit tests."""

from __future__ import annotations

import json

from convmerge.adapters.alpaca import iter_from_alpaca_line
from convmerge.adapters.sharegpt import iter_from_sharegpt_line


def test_alpaca_basic() -> None:
    rec = {
        "instruction": "Hello",
        "input": "context",
        "output": "Hi there",
    }
    examples = list(iter_from_alpaca_line(rec))
    assert len(examples) == 1
    assert [m.role for m in examples[0].messages] == ["user", "assistant"]
    assert "Hello" in examples[0].messages[0].content
    assert "context" in examples[0].messages[0].content
    assert examples[0].messages[1].content == "Hi there"


def test_alpaca_instruction_only() -> None:
    rec = {"instruction": "Q", "output": "A"}
    examples = list(iter_from_alpaca_line(rec))
    assert len(examples) == 1
    assert examples[0].messages[0].content == "Q"
    assert examples[0].messages[1].content == "A"


def test_sharegpt_pair() -> None:
    rec = {
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "answer"},
        ]
    }
    examples = list(iter_from_sharegpt_line(rec))
    assert len(examples) == 1
    assert examples[0].messages[0].role == "user"
    assert examples[0].messages[1].role == "assistant"


def test_sharegpt_fixture_file() -> None:
    from pathlib import Path

    p = Path(__file__).parent / "fixtures" / "sharegpt_one.jsonl"
    line = p.read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    examples = list(iter_from_sharegpt_line(rec))
    assert len(examples) >= 1
