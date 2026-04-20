"""Tests for convmerge.normalize.schema."""

from __future__ import annotations

from pathlib import Path

from convmerge.normalize.schema import is_uniform_schema, key_frequency


def test_is_uniform_schema_true() -> None:
    records = [{"a": 1, "b": 2}, {"a": 10, "b": 20}]
    assert is_uniform_schema(records) is True


def test_is_uniform_schema_false_on_different_keys() -> None:
    records = [{"a": 1, "b": 2}, {"a": 10, "c": 20}]
    assert is_uniform_schema(records) is False


def test_is_uniform_schema_false_on_non_dict() -> None:
    records = [{"a": 1}, "not a dict"]
    assert is_uniform_schema(records) is False  # type: ignore[list-item]


def test_key_frequency_counts_top_level() -> None:
    records = [{"a": 1, "b": 2}, {"a": 10}, {"b": 5, "c": 99}]
    counts = key_frequency(records)
    assert counts == {"a": 2, "b": 2, "c": 1}


def test_key_frequency_recursive() -> None:
    records = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    ]
    counts = key_frequency(records, recursive=True)
    assert counts["messages"] == 1
    assert counts["role"] == 2
    assert counts["content"] == 2


def test_is_uniform_schema_from_jsonl_file(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text('{"x":1,"y":2}\n{"x":3,"y":4}\n', encoding="utf-8")
    assert is_uniform_schema(p) is True
