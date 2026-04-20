"""Tests for convmerge.normalize.convert_turns."""

from __future__ import annotations

from convmerge.normalize.convert_turns import (
    multi_turn_to_single_turn_record,
    single_turn_to_multi_turn_record,
)


def test_single_to_multi_basic() -> None:
    r = single_turn_to_multi_turn_record({"instruction": "Q", "input": "ctx", "output": "A"})
    assert r is not None
    assert r["messages"][0]["role"] == "user"
    assert "Q" in r["messages"][0]["content"]
    assert "ctx" in r["messages"][0]["content"]
    assert r["messages"][1] == {"role": "assistant", "content": "A"}


def test_single_to_multi_requires_output() -> None:
    assert single_turn_to_multi_turn_record({"instruction": "Q"}) is None


def test_multi_to_single_flattens_users() -> None:
    sample = {
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
    }
    r = multi_turn_to_single_turn_record(sample)
    assert r is not None
    assert r["instruction"] == "u1\nu2"
    assert r["output"] == "a2"
    assert r["input"] == ""


def test_multi_to_single_returns_none_without_assistant() -> None:
    sample = {"messages": [{"role": "user", "content": "u1"}]}
    assert multi_turn_to_single_turn_record(sample) is None
