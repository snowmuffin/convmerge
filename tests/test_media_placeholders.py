"""--media placeholders: the TRL vision-language layout (1.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge import EmitOptions, convert_file, validate_file
from convmerge.arrow import type_conflicts

ROWS = [
    {"id": "1", "image": "imgs/a.png", "conversations": [
        {"from": "human", "value": "<image>\nWhat color is this?"},
        {"from": "gpt", "value": "Red."},
        {"from": "human", "value": "Sure?"}, {"from": "gpt", "value": "Yes."}]},
    {"id": "2", "conversations": [{"from": "human", "value": "Hi"},
                                  {"from": "gpt", "value": "Hello."}]},
    {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        {"type": "text", "text": "Describe"}, {"type": "image", "image": "b.png"}]},
        {"role": "assistant", "content": "Two pictures."}]},
    {"messages": [{"role": "user", "content": "Weather?"},
                  {"role": "assistant", "content": None, "tool_calls": [
                      {"type": "function", "function": {"name": "w", "arguments": "{}"}}]},
                  {"role": "tool", "content": "sunny"},
                  {"role": "assistant", "content": "Sunny."}]},
]  # fmt: skip


def _convert(tmp_path: Path, **options) -> list[dict]:
    src = tmp_path / "in.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    convert_file(src, out, adapter_name="auto", output_format="messages",
                 emit_options=EmitOptions(**options))  # fmt: skip
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_placeholders_layout(tmp_path: Path) -> None:
    rows = _convert(tmp_path, media="placeholders")
    assert rows[0] == {
        "messages": [
            {"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": "What color is this?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Red."}]},
            {"role": "user", "content": [{"type": "text", "text": "Sure?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Yes."}]},
        ],
        "images": ["imgs/a.png"],
    }  # fmt: skip
    assert rows[1]["images"] == []  # every row has the column
    assert rows[2]["messages"][0]["content"] == [
        {"type": "image"}, {"type": "text", "text": "Describe"}, {"type": "image"}]  # fmt: skip
    assert rows[2]["images"] == ["https://x/y.png", "b.png"]  # in order of the placeholders
    call = rows[3]["messages"][1]
    assert call["content"] == [] and call["tool_calls"][0]["function"]["name"] == "w"
    # every content is a list of parts: one Arrow type for the column
    assert type_conflicts(tmp_path / "out.jsonl") == []


def test_urls_default_is_unchanged(tmp_path: Path) -> None:
    rows = _convert(tmp_path)
    assert "images" not in rows[0]
    assert rows[0]["messages"][0]["content"][0] == {
        "type": "image_url", "image_url": {"url": "imgs/a.png"}}  # fmt: skip
    assert rows[1]["messages"][0]["content"] == "Hi"


def test_placeholders_read_back_and_validate(tmp_path: Path) -> None:
    rows = _convert(tmp_path, media="placeholders")
    out = tmp_path / "out.jsonl"
    again = tmp_path / "again.jsonl"
    convert_file(out, again, adapter_name="auto", output_format="messages",
                 emit_options=EmitOptions(media="placeholders"))  # fmt: skip
    assert again.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")
    assert validate_file(out).dropped == 0
    rows[0]["images"] = []  # a placeholder without its image
    rows[2]["images"].append("extra.png")  # an image without a placeholder
    bad = tmp_path / "bad.jsonl"
    bad.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert validate_file(bad).dropped == 2


def test_placeholders_options_and_errors(tmp_path: Path, capsys) -> None:
    from convmerge.cli import main
    from convmerge.config import emit_options_from_mapping
    from convmerge.recipe.schema import parse_recipe

    with pytest.raises(ValueError, match="media must be"):
        EmitOptions(media="inline")
    assert emit_options_from_mapping({"media": "placeholders"}).media == "placeholders"
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(ROWS[0]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        main(["convert", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--from", "auto",
              "--format", "sharegpt", "--media", "placeholders"])  # fmt: skip
    assert e.value.code == 2 and "layout of the messages format" in capsys.readouterr().err
    main(["convert", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--from", "auto",
          "--media", "placeholders"])  # fmt: skip
    assert json.loads((tmp_path / "o.jsonl").read_text())["images"] == ["imgs/a.png"]
    recipe = parse_recipe({"version": 1, "output": "o.jsonl", "sources": {"a": {
        "path": "in.jsonl", "convert": {"from": "auto", "media": "placeholders"}}}},
        path=tmp_path / "r.yaml")  # fmt: skip
    assert recipe.sources["a"].convert.emit["media"] == "placeholders"
