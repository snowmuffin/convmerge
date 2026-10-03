"""Output-format options: alpaca multi-turn policy, system field, keep-meta, presets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.config import build_convert_config
from convmerge.emitters import (
    EmitOptions,
    UnrepresentableExample,
    emit_alpaca,
    emit_messages,
    get_emitter,
)
from convmerge.models import ChatMessage, ContentPart, ToolCall, TrainingExample
from convmerge.preset import load_convert_preset

S = ChatMessage("system", "sys")
U1, A1 = ChatMessage("user", "q1"), ChatMessage("assistant", "a1")
U2, A2 = ChatMessage("user", "q2"), ChatMessage("assistant", "a2")


def _ex(*msgs: ChatMessage, **kw) -> TrainingExample:
    return TrainingExample(messages=list(msgs), **kw)


def test_alpaca_single_pair_with_system() -> None:
    assert emit_alpaca(_ex(S, U1, A1)) == {
        "instruction": "q1",
        "input": "",
        "output": "a1",
        "system": "sys",
    }


def test_alpaca_flatten_records_lossy_note() -> None:
    notes: list[str] = []
    row = emit_alpaca(_ex(U1, A1, U2, A2), notes=notes)
    assert row == {"instruction": "q1\nq2", "input": "", "output": "a2"}
    assert notes == ["lossy_multiturn_flattened"]


def test_alpaca_history_and_drop() -> None:
    hist = EmitOptions(alpaca_multiturn="history")
    assert emit_alpaca(_ex(S, U1, A1, U2, A2), options=hist) == {
        "instruction": "q2",
        "input": "",
        "output": "a2",
        "history": [["q1", "a1"]],
        "system": "sys",
    }
    with pytest.raises(UnrepresentableExample) as exc:
        emit_alpaca(_ex(U1, U2, A2), options=hist)
    assert exc.value.reason == "unrepresentable_multiturn"
    with pytest.raises(UnrepresentableExample):
        emit_alpaca(_ex(U1, A1, U2, A2), options=EmitOptions(alpaca_multiturn="drop"))


@pytest.mark.parametrize(
    ("example", "reason"),
    [
        (
            _ex(U1, ChatMessage("assistant", None, tool_calls=[ToolCall("f")])),
            "unrepresentable_tool_calls",
        ),
        (
            _ex(U1, A1, tools=[{"type": "function", "function": {"name": "f"}}]),
            "unrepresentable_tool_calls",
        ),
        (
            _ex(ChatMessage("user", [ContentPart("image", url="x.png")]), A1),
            "unrepresentable_media",
        ),
    ],
)
def test_alpaca_rejects_tools_and_media(example: TrainingExample, reason: str) -> None:
    with pytest.raises(UnrepresentableExample) as exc:
        emit_alpaca(example)
    assert exc.value.reason == reason


def test_keep_meta_all_subset_and_key() -> None:
    ex = _ex(U1, A1, meta={"source": "chat", "id": 7})
    assert "meta" not in emit_messages(ex)
    assert emit_messages(ex, options=EmitOptions(keep_meta=True))["meta"] == {
        "source": "chat",
        "id": 7,
    }
    assert emit_messages(ex, options=EmitOptions(keep_meta=["id"], meta_key="_m"))["_m"] == {
        "id": 7
    }
    assert "meta" not in emit_messages(ex, options=EmitOptions(keep_meta=["missing"]))
    assert emit_alpaca(ex, options=EmitOptions(keep_meta=True))["meta"] == {
        "source": "chat",
        "id": 7,
    }


def test_emit_options_validation() -> None:
    with pytest.raises(ValueError, match="alpaca_multiturn"):
        EmitOptions(alpaca_multiturn="merge")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown output format"):
        get_emitter("no-such-format")


def test_preset_output_options_and_cli_override(tmp_path: Path) -> None:
    preset = tmp_path / "p.json"
    preset.write_text(
        json.dumps(
            {
                "adapter": "chat",
                "output_format": "alpaca",
                "output_options": {"alpaca_multiturn": "history", "keep_meta": ["id"]},
            }
        ),
        encoding="utf-8",
    )
    assert load_convert_preset(preset).emit_options == EmitOptions(
        alpaca_multiturn="history", keep_meta=("id",)
    )
    cfg = build_convert_config(preset_path=preset, emit_overrides={"alpaca_multiturn": "drop"})
    assert cfg.emit_options == EmitOptions(alpaca_multiturn="drop", keep_meta=("id",))

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"adapter": "chat", "output_format": "messages", "output_options": {"x": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="output_options"):
        load_convert_preset(bad)


def test_cli_keep_meta_and_alpaca_flags(tmp_path: Path, capsys) -> None:
    src = tmp_path / "in.jsonl"
    rec = {
        "id": "r1",
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ],
    }
    src.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    out = tmp_path / "o.jsonl"
    base = ["convert", "-i", str(src), "-o", str(out), "--from", "chat"]

    main([*base, "-f", "messages", "--keep-meta", "id"])
    assert json.loads(out.read_text())["meta"] == {"id": "r1"}

    main([*base, "-f", "alpaca"])
    assert "1 example written lossily: lossy_multiturn_flattened" in capsys.readouterr().err

    main([*base, "-f", "alpaca", "--alpaca-multiturn", "history", "--keep-meta"])
    row = json.loads(out.read_text())
    assert row["history"] == [["q1", "a1"]]
    assert row["meta"] == {"source": "chat", "id": "r1"}
