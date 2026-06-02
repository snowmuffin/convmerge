"""Tests for convmerge.normalize.schema."""

from __future__ import annotations

from pathlib import Path

from convmerge.normalize.schema import (
    is_uniform_schema,
    key_frequency,
    profile_schema,
)


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


# --- profile_schema --------------------------------------------------------


def test_profile_schema_top_level_types_and_presence() -> None:
    records = [
        {"instruction": "Say hello", "output": "Hi"},
        {"instruction": "Bye", "output": "See ya", "extra": 1},
    ]
    report = profile_schema(records)
    assert report["records"] == 2
    assert report["uniform_top_level"] is False
    fields = report["fields"]
    assert fields["instruction"]["types"] == {"str": 2}
    assert fields["instruction"]["present"] == 2
    assert fields["instruction"]["presence"] == 1.0
    # Optional field present in only one record.
    assert fields["extra"]["present"] == 1
    assert fields["extra"]["presence"] == 0.5
    assert fields["extra"]["types"] == {"int": 1}
    assert "Say hello" in fields["instruction"]["examples"]


def test_profile_schema_preserves_nesting_path() -> None:
    records = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    ]
    report = profile_schema(records)
    msgs = report["fields"]["messages"]
    assert msgs["types"] == {"array": 1}
    # Inner fields are reported under "items", not flattened to top level.
    assert set(msgs["items"].keys()) == {"role", "content"}
    assert msgs["items"]["role"]["types"] == {"str": 2}
    assert "user" in msgs["items"]["role"]["examples"]
    # The nested role/content must NOT leak into the top-level field set.
    assert set(report["fields"].keys()) == {"messages"}


def test_profile_schema_object_field_uses_fields_key() -> None:
    records = [{"meta": {"source": "x", "id": 1}}]
    report = profile_schema(records)
    meta = report["fields"]["meta"]
    assert meta["types"] == {"object": 1}
    assert set(meta["fields"].keys()) == {"source", "id"}


def test_profile_schema_example_truncation() -> None:
    long = "z" * 200
    report = profile_schema([{"t": long}], max_examples=1)
    ex = report["fields"]["t"]["examples"][0]
    assert ex.endswith("...")
    assert len(ex) == 80


def test_profile_schema_from_file(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
    report = profile_schema(p, max_rows=1)
    assert report["records"] == 1
