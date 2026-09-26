"""JSONL whose lines are arrays (#26) and element profiling of list fields (#27)."""

from __future__ import annotations

import json
from pathlib import Path

from convmerge.cli import main
from convmerge.normalize.jsonl import detect_jsonl_shape, iter_json_records, normalize_to_jsonl
from convmerge.normalize.schema import profile_schema

TURNS_1 = [
    {"speaker": "client", "utterance": "I can't sleep."},
    {"speaker": "counselor", "utterance": "How long has it been?"},
]
TURNS_2 = [
    {"speaker": "client", "utterance": "Work is hard."},
    {"speaker": "counselor", "utterance": "Tell me more."},
]


def _arrays_file(path: Path) -> Path:
    path.write_text(json.dumps(TURNS_1) + "\n" + json.dumps(TURNS_2) + "\n", encoding="utf-8")
    return path


def test_detects_jsonl_of_arrays_but_not_single_or_pretty_arrays(tmp_path: Path) -> None:
    assert detect_jsonl_shape(_arrays_file(tmp_path / "a.jsonl")) == "jsonl_of_arrays"
    one = tmp_path / "one.json"
    one.write_text(json.dumps(TURNS_1) + "\n\n", encoding="utf-8")
    assert detect_jsonl_shape(one) == "json_array"
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps([{"a": 1}, {"a": 2}], indent=2), encoding="utf-8")
    assert detect_jsonl_shape(pretty) == "json_array"


def test_normalize_wraps_array_records(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    assert normalize_to_jsonl(_arrays_file(tmp_path / "a.jsonl"), out) == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows == [{"conversation": TURNS_1}, {"conversation": TURNS_2}]

    custom = tmp_path / "custom.jsonl"
    normalize_to_jsonl(tmp_path / "a.jsonl", custom, array_key="turns")
    assert json.loads(custom.read_text().splitlines()[0]) == {"turns": TURNS_1}


def test_json_array_of_arrays_is_wrapped(tmp_path: Path) -> None:
    src = tmp_path / "a.json"
    src.write_text(json.dumps([TURNS_1, {"k": 1}]), encoding="utf-8")
    out = tmp_path / "o.jsonl"
    normalize_to_jsonl(src, out)
    assert [json.loads(x) for x in out.read_text().splitlines()] == [
        {"conversation": TURNS_1},
        {"k": 1},
    ]


def test_inspect_profiles_array_records(tmp_path: Path) -> None:
    p = _arrays_file(tmp_path / "a.jsonl")
    assert len(list(iter_json_records(p))) == 2
    assert list(iter_json_records(p, array_key=None)) == []
    report = profile_schema(p)
    assert report["records"] == 2
    items = report["fields"]["conversation"]["items"]
    assert set(items) == {"speaker", "utterance"}


def test_arrays_end_to_end_with_preset(tmp_path: Path) -> None:
    norm = tmp_path / "norm.jsonl"
    main(["normalize", "-i", str(_arrays_file(tmp_path / "a.jsonl")), "-o", str(norm)])
    preset = tmp_path / "p.json"
    preset.write_text(
        json.dumps(
            {
                "adapter": "chat",
                "output_format": "messages",
                "adapter_options": {
                    "chat": {
                        "role_keys": ["speaker"],
                        "content_keys": ["utterance"],
                        "role_map": {"client": "user", "counselor": "assistant"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    main(["convert", "-i", str(norm), "-o", str(out), "--preset", str(preset)])
    first = json.loads(out.read_text().splitlines()[0])
    assert first == {
        "messages": [
            {"role": "user", "content": "I can't sleep."},
            {"role": "assistant", "content": "How long has it been?"},
        ]
    }


def test_profile_reports_element_types_and_examples() -> None:
    records = [
        {
            "conversations": ["hi", "hello", "bye"],
            "tags": [1, "a", None, [2]],
            "msgs": [{"r": "u"}],
        },
        {"conversations": ["hi again", "yo"]},
    ]
    fields = profile_schema(records, max_examples=2)["fields"]
    conv = fields["conversations"]
    assert conv["element_types"] == {"str": 5}
    assert conv["element_examples"] == ["hi", "hello"]
    assert "items" not in conv
    assert fields["tags"]["element_types"] == {"int": 1, "str": 1, "null": 1, "array": 1}
    assert fields["tags"]["element_examples"] == [1, "a"]
    msgs = fields["msgs"]
    assert msgs["element_types"] == {"object": 1}
    assert set(msgs["items"]) == {"r"}
