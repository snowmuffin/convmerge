"""Tests for convmerge.normalize.reshape and the alpaca remap helper."""

from __future__ import annotations

import json
from pathlib import Path

from convmerge.adapters.alpaca import remap_to_alpaca
from convmerge.normalize.reshape import (
    classify_row_shape,
    parse_tagged_text,
    unify_alpaca_dir,
    unify_message_entries,
    unify_messages_dir,
)


def test_classify_row_shape_variants() -> None:
    assert classify_row_shape({"messages": [{"role": "user", "content": "x"}]}) == "chat"
    assert (
        classify_row_shape({"conversation_a": [{"role": "user"}], "conversation_b": []})
        == "pairwise"
    )
    assert classify_row_shape({"instruction": "a", "output": "b"}) == "alpaca"
    assert classify_row_shape({"text": "just text"}) == "plain_text"
    assert classify_row_shape({"misc": 1}) == "unknown"


def test_parse_tagged_text_default_map() -> None:
    msgs = parse_tagged_text("<sys>be nice<usr>hi<bot>hello")
    assert msgs == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_parse_tagged_text_custom_map() -> None:
    msgs = parse_tagged_text("<H>hi<A>hey", tag_map={"h": "user", "a": "assistant"})
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]


def test_remap_to_alpaca_picks_first_hit() -> None:
    out = remap_to_alpaca(
        {"question": "q", "answer": "a", "meta": "drop", "input": "ctx"},
        drop_keys=("meta",),
    )
    assert out == {"instruction": "q", "input": "ctx", "output": "a"}


def test_remap_to_alpaca_returns_none_when_incomplete() -> None:
    assert remap_to_alpaca({"question": "only q"}) is None


def test_unify_messages_dir_hoists_conversations(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jsonl").write_text(
        json.dumps(
            {
                "conversation": [
                    {"from": "human", "value": "q"},
                    {"from": "gpt", "value": "a"},
                ]
            }
        )
        + "\n"
    )
    dst = tmp_path / "dst"
    n_files, n_rows = unify_messages_dir(src, dst)
    assert n_files == 1
    assert n_rows == 1
    obj = json.loads((dst / "a.jsonl").read_text())
    assert obj["messages"] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_unify_messages_dir_in_place(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"}]}) + "\n"
    )
    n_files, _ = unify_messages_dir(src, src)
    assert n_files == 1
    obj = json.loads((src / "a.jsonl").read_text())
    assert obj["messages"] == [{"role": "user", "content": "q"}]


def test_unify_message_entries_maps_from_to_role(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"from": "human", "content": "q"},
                    {"from": "gpt", "content": "a"},
                ]
            }
        )
        + "\n"
    )
    dst = tmp_path / "dst"
    unify_message_entries(src, dst, source_keys=("from",), target_key="role")
    obj = json.loads((dst / "a.jsonl").read_text())
    assert obj["messages"] == [
        {"role": "human", "content": "q"},
        {"role": "gpt", "content": "a"},
    ]


def test_unify_alpaca_dir_writes_standard_schema(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jsonl").write_text(
        json.dumps({"prompt": "q1", "response": "a1", "id": "drop-me"})
        + "\n"
        + json.dumps({"missing": "instr"})
        + "\n"
    )
    dst = tmp_path / "dst"
    n_files, n_rows = unify_alpaca_dir(
        src,
        dst,
        drop_keys=("id",),
    )
    assert n_files == 1
    assert n_rows == 1
    obj = json.loads((dst / "a.jsonl").read_text().strip())
    assert obj == {"instruction": "q1", "input": "", "output": "a1"}
