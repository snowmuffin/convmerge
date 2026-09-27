"""Fixes for strict chat templates: system fold/drop, merge, split turns, last-turn reasoning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.config import build_convert_config
from convmerge.convert import ConvertStats, convert_file, convert_with_config
from convmerge.models import ChatMessage, ContentPart, ToolCall, TrainingExample
from convmerge.preset import load_convert_preset
from convmerge.recipe import RecipeError, parse_recipe
from convmerge.transforms import TransformOptions, apply_transforms

S = ChatMessage("system", "be brief")
U1, U2, U3 = ChatMessage("user", "q1"), ChatMessage("user", "q2"), ChatMessage("user", "q3")
A1 = ChatMessage("assistant", "a1", reasoning="r1")
A2 = ChatMessage("assistant", "<think>\nr2\n</think>\n\na2")
A3 = ChatMessage("assistant", "a3", reasoning="r3")


def _run(msgs: list[ChatMessage], **opts) -> tuple[list[TrainingExample], dict[str, int]]:
    counts: dict[str, int] = {}
    out = apply_transforms(TrainingExample(messages=msgs), TransformOptions(**opts), counts)
    return out, counts


def test_defaults_change_nothing() -> None:
    assert not TransformOptions().active
    msgs = [S, U1, U2, A1]
    (ex,), counts = _run(msgs)
    assert ex.messages == msgs and counts == {}


def test_system_fold_and_drop() -> None:
    (ex,), counts = _run([S, ChatMessage("system", "and kind"), U1, A1], system="fold")
    assert ex.messages[0] == ChatMessage("user", "be brief\n\nand kind\n\nq1")
    assert counts == {"system_folded": 2}
    img = ContentPart("image", url="a.png")
    (ex,), _ = _run([S, ChatMessage("user", (img,)), A1], system="fold")
    assert ex.messages[0].content == (ContentPart("text", text="be brief"), img)
    (ex,), counts = _run([S, U1, A1, S, U2, A3], system="drop")
    assert [m.role for m in ex.messages] == ["user", "assistant", "user", "assistant"]
    assert counts == {"system_dropped": 2}
    (ex,), counts = _run([S], system="fold")  # nothing to fold into
    assert ex.messages == [S] and counts == {}


def test_merge_consecutive() -> None:
    (ex,), counts = _run([U1, U2, A1, A3], merge_consecutive=True)
    assert ex.messages == [
        ChatMessage("user", "q1\n\nq2"),
        ChatMessage("assistant", "a1\n\na3", reasoning="r1\n\nr3"),
    ]
    assert counts == {"turns_merged": 2}
    call = ChatMessage("assistant", None, tool_calls=(ToolCall("f"),))
    tool = ChatMessage("tool", "x")
    alice, bob = ChatMessage("user", "hi", name="alice"), ChatMessage("user", "yo", name="bob")
    msgs = [alice, bob, call, tool, tool, ChatMessage("assistant", "done")]
    (ex,), counts = _run(msgs, merge_consecutive=True)
    assert ex.messages == msgs and counts == {}


def test_split_turns() -> None:
    out, counts = _run([S, U1, A1, U2, A2, U3], split_turns=True)
    assert counts == {"split_examples": 2}
    first, second = out
    assert first.messages == [S, U1, A1] and first.meta == {"turn": 0}
    # Earlier answers stay, their reasoning goes; the target turn keeps its own.
    assert second.messages == [S, U1, ChatMessage("assistant", "a1"), U2, A2]
    assert second.meta == {"turn": 1}
    (single,), counts = _run([U1, A1], split_turns=True)
    assert single.messages == [U1, A1] and counts == {}


def test_split_turns_keeps_tool_rounds_together() -> None:
    call = ChatMessage("assistant", None, tool_calls=(ToolCall("f"),))
    msgs = [U1, call, ChatMessage("tool", "x"), A1, U2, A3]
    first, second = _run(msgs, split_turns=True)[0]
    assert first.messages == msgs[:4]
    assert second.messages[-2:] == [U2, A3]


def test_reasoning_turns_last() -> None:
    (ex,), counts = _run([U1, A1, U2, A2, U3, A3], reasoning_turns="last")
    assert ex.messages == [
        U1, ChatMessage("assistant", "a1"), U2, ChatMessage("assistant", "a2"), U3, A3,
    ]  # fmt: skip
    assert counts == {"reasoning_stripped": 2}


def test_pairs_get_the_same_fixes_on_both_sides() -> None:
    no = ChatMessage("assistant", "no")
    ex = TrainingExample(messages=[S, U1, A1, U2, A3], rejected=[S, U1, A1, U2, no])
    (out,) = apply_transforms(ex, TransformOptions(system="fold", reasoning_turns="last"))
    assert out.messages[:2] == out.rejected[:2]
    assert out.messages[1] == ChatMessage("assistant", "a1")


def test_bad_values() -> None:
    with pytest.raises(ValueError, match="system must be"):
        TransformOptions(system="merge")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reasoning_turns"):
        TransformOptions(reasoning_turns="first")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot split preference pairs"):
        build_convert_config(
            adapter="auto", output_format="preference", transform_overrides={"split_turns": True}
        )


ROWS = [
    {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"},
                  {"role": "user", "content": "more"}, {"role": "assistant", "content": "a1"},
                  {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]},
    {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
] * 3  # fmt: skip


def _src(tmp_path: Path) -> Path:
    p = tmp_path / "in.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    return p


def test_convert_reports_fixes_and_workers_match(tmp_path: Path) -> None:
    opts = TransformOptions(system="fold", merge_consecutive=True, split_turns=True)
    single, parallel = ConvertStats(), ConvertStats()
    kw = {"adapter_name": "auto", "output_format": "messages", "transform_options": opts}
    convert_file(_src(tmp_path), tmp_path / "a.jsonl", stats=single, **kw)
    convert_file(_src(tmp_path), tmp_path / "b.jsonl", stats=parallel, workers=2, **kw)
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert single.transforms == parallel.transforms == {
        "system_folded": 3, "turns_merged": 3, "split_examples": 6,
    }  # fmt: skip
    assert single.written == 9
    assert single.to_report()["transforms"] == single.transforms


def test_cli_flags_and_preset(tmp_path: Path, capsys) -> None:
    out = tmp_path / "o.jsonl"
    main(["convert", "-i", str(_src(tmp_path)), "-o", str(out), "--from", "auto",
          "--format", "sharegpt", "--system", "fold", "--merge-consecutive"])  # fmt: skip
    err = capsys.readouterr().err
    assert "fixes: system_folded=3, turns_merged=3" in err
    first = json.loads(out.read_text().splitlines()[0])
    assert first["conversations"][0] == {"from": "human", "value": "s\n\nq1\n\nmore"}

    preset = tmp_path / "p.json"
    preset.write_text(json.dumps({"adapter": "auto", "output_format": "messages",
                                  "transforms": {"split_turns": True}}))  # fmt: skip
    cfg = load_convert_preset(preset)
    assert cfg.transform_options == TransformOptions(split_turns=True)
    st = ConvertStats()
    convert_with_config(_src(tmp_path), out, cfg, stats=st)
    assert st.transforms == {"split_examples": 6}

    with pytest.raises(SystemExit) as e:
        main(["convert", "-i", str(_src(tmp_path)), "-o", str(out), "--from", "auto",
              "--format", "preference", "--split-turns"])  # fmt: skip
    assert e.value.code == 2


def test_recipe_source_keys(tmp_path: Path) -> None:
    _src(tmp_path)
    conv = {"from": "auto", "system": "fold", "merge_consecutive": True, "reasoning": "inline"}
    recipe = parse_recipe(
        {"output": "o.jsonl", "sources": {"a": {"path": "in.jsonl", "convert": conv}}},
        path=tmp_path / "r.yaml",
    )
    spec = recipe.sources["a"].convert
    assert spec.transforms == {"system": "fold", "merge_consecutive": True}
    assert spec.options(tmp_path)["transforms"] == spec.transforms
    assert spec.emit == {"reasoning": "inline"}
    plain = parse_recipe(
        {"output": "o.jsonl", "sources": {"a": {"path": "in.jsonl", "convert": {"from": "auto"}}}},
        path=tmp_path / "r.yaml",
    )
    assert "transforms" not in plain.sources["a"].convert.options(tmp_path)  # old locks stay valid
    for bad in ({"system": "sideways"}, {"merge_consecutive": "yes"}, {"reasoning": "x"}):
        with pytest.raises(RecipeError):
            parse_recipe(
                {"output": "o.jsonl",
                 "sources": {"a": {"path": "in.jsonl", "convert": {"from": "auto", **bad}}}},
                path=tmp_path / "r.yaml",
            )  # fmt: skip
