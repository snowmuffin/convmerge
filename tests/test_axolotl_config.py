"""convmerge axolotl-config: the datasets: block matching a converted file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.axolotl import dataset_config, render_config
from convmerge.cli import main

U = {"role": "user", "content": "q"}
A = {"role": "assistant", "content": "a"}


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_messages(tmp_path: Path) -> None:
    entry = dataset_config(
        _write(tmp_path / "m.jsonl", [{"messages": [U, A]}]), file_path="m.jsonl"
    )
    assert entry == {
        "path": "m.jsonl",
        "ds_type": "json",
        "type": "chat_template",
        "field_messages": "messages",
        "roles_to_train": ["assistant"],
    }
    thinking = [{"messages": [U, {**A, "thinking": "t"}]}]
    entry = dataset_config(_write(tmp_path / "t.jsonl", thinking))
    assert (entry["field_thinking"], entry["template_thinking_key"]) == ("thinking", "thinking")


def test_sharegpt(tmp_path: Path) -> None:
    rows = [{"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
             "system": "s"}]  # fmt: skip
    entry = dataset_config(_write(tmp_path / "s.jsonl", rows))
    assert entry["message_property_mappings"] == {"role": "from", "content": "value"}
    assert entry["roles"]["user"] == ["human", "user"] and entry["field_system"] == "system"
    rows[0]["conversations"].insert(1, {"from": "function_call", "value": "{}"})
    with pytest.raises(ValueError, match="--format messages"):
        dataset_config(_write(tmp_path / "f.jsonl", rows))


def test_preference_pairs(tmp_path: Path) -> None:
    pair = {"prompt": [U], "chosen": [A], "rejected": [{**A, "content": "b"}]}
    entry = dataset_config(_write(tmp_path / "p.jsonl", [pair]))
    assert entry["type"] == "chat_template.default" and entry["_rl"] == "dpo"
    assert (entry["field_messages"], entry["field_chosen"]) == ("prompt", "chosen")
    long = {**pair, "chosen": [A, U, A]}
    with pytest.raises(ValueError, match="more than one message"):
        dataset_config(_write(tmp_path / "l.jsonl", [long]))
    sg = {
        "conversations": [{"from": "human", "value": "q"}],
        "chosen": {"from": "gpt", "value": "a"},
        "rejected": {"from": "gpt", "value": "b"},
    }
    entry = dataset_config(_write(tmp_path / "sg.jsonl", [sg]))
    assert entry["field_messages"] == "conversations" and "message_property_mappings" in entry


def test_alpaca_and_unknown(tmp_path: Path) -> None:
    row = {"instruction": "q", "input": "", "output": "a"}
    assert dataset_config(_write(tmp_path / "a.jsonl", [row]))["type"] == "alpaca"
    with pytest.raises(ValueError, match="not a messages"):
        dataset_config(_write(tmp_path / "x.jsonl", [{"text": "hi"}]))
    with pytest.raises(ValueError, match="no JSON object rows"):
        dataset_config(_write(tmp_path / "e.jsonl", []))


def test_render_and_cli(tmp_path: Path, capsys) -> None:
    (tmp_path / "data").mkdir()
    train = _write(tmp_path / "data" / "train.jsonl", [{"messages": [U, A]}])
    val = _write(tmp_path / "data" / "val.jsonl", [{"messages": [U, A]}])
    main(["axolotl-config", "-i", str(train), "--val", str(val), "--config-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.splitlines()[1:] == [
        "datasets:",
        '  - path: "data/train.jsonl"',
        '    ds_type: "json"',
        '    type: "chat_template"',
        '    field_messages: "messages"',
        '    roles_to_train: ["assistant"]',
        "test_datasets:",
        '  - path: "data/val.jsonl"',
        '    ds_type: "json"',
        '    type: "chat_template"',
        '    field_messages: "messages"',
        '    roles_to_train: ["assistant"]',
        '    split: "train"',
    ]
    rejected = {**A, "content": "b"}
    pair = _write(tmp_path / "p.jsonl", [{"prompt": [U], "chosen": [A], "rejected": [rejected]}])
    text = render_config(dataset_config(pair))
    assert "\nrl: dpo\n" in text and "_rl" not in text
    with pytest.raises(SystemExit) as e:
        main(["axolotl-config", "-i", str(tmp_path / "missing.jsonl")])
    assert e.value.code == 1


def test_yaml_parses(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    rows = [{"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}]
    text = render_config(dataset_config(_write(tmp_path / "s.jsonl", rows), file_path="s.jsonl"))
    data = yaml.safe_load(text)
    assert data["datasets"][0]["roles"]["assistant"] == ["gpt", "assistant"]
