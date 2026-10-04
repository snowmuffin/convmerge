"""Reasoning traces: reading them, placing them (--reasoning), and tool-call content."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.adapters.chat import iter_from_chat_line
from convmerge.cli import main
from convmerge.config import build_convert_config
from convmerge.convert import ConvertStats, convert_file
from convmerge.emitters import (
    EmitOptions,
    emit_alpaca,
    emit_messages,
    emit_preference,
    emit_sharegpt,
)
from convmerge.llamafactory import dataset_info_entry
from convmerge.models import ChatMessage, ContentPart, ToolCall, TrainingExample
from convmerge.reasoning import (
    join_inline,
    reasoning_text,
    split_inline,
    strip_reasoning,
    to_field,
    to_inline,
)

INLINE = "<think>\n2+2=4\n</think>\nThe answer is 4."
Q = {"role": "user", "content": "q"}


def _one(record: dict, **kw) -> TrainingExample:
    (example,) = list(iter_from_chat_line(record, **kw))
    return example


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- helpers -----------------------------------------------------------------


def test_split_and_join_inline() -> None:
    assert split_inline(INLINE) == ("2+2=4", "The answer is 4.")
    assert split_inline("  <think>a\nb</think>\n\n\nans") == ("a\nb", "ans")
    assert split_inline("no think") == (None, "no think")
    assert split_inline("<think>never closed") == (None, "<think>never closed")
    assert split_inline("answer <think>late</think>") == (None, "answer <think>late</think>")
    assert join_inline("r", "a") == "<think>\nr\n</think>\n\na"


def test_move_between_field_and_inline() -> None:
    m = ChatMessage("assistant", INLINE)
    field = to_field(m)
    assert (field.content, field.reasoning) == ("The answer is 4.", "2+2=4")
    assert to_inline(field).content == "<think>\n2+2=4\n</think>\n\nThe answer is 4."
    assert to_inline(field).reasoning is None
    assert reasoning_text(m) == reasoning_text(field) == "2+2=4"
    assert strip_reasoning(m) == ChatMessage("assistant", "The answer is 4.")
    assert strip_reasoning(field) == ChatMessage("assistant", "The answer is 4.")
    user = ChatMessage("user", INLINE)
    assert to_field(user) is user and strip_reasoning(user) is user


def test_parts_content_keeps_media() -> None:
    img = ContentPart("image", url="a.png")
    m = ChatMessage("assistant", (ContentPart("text", text=INLINE), img))
    field = to_field(m)
    assert field.reasoning == "2+2=4"
    assert field.content == (ContentPart("text", text="The answer is 4."), img)
    only_media = ChatMessage("assistant", (img,), reasoning="r")
    assert to_inline(only_media).content[0].text.startswith("<think>\nr\n</think>")


# --- reading -------------------------------------------------------------------


@pytest.mark.parametrize("key", ["reasoning_content", "thinking", "reasoning"])
def test_turn_reasoning_keys(key: str) -> None:
    ex = _one(
        {
            "messages": [
                {"role": "user", "content": "q", key: "ignored on user turns"},
                {"role": "assistant", "content": "a", key: "because"},
            ]
        }
    )
    assert ex.messages[0].reasoning is None
    assert ex.messages[1].reasoning == "because"


def test_inline_think_stays_in_content_by_default() -> None:
    ex = _one({"messages": [Q, {"role": "assistant", "content": INLINE}]})
    assert ex.messages[1] == ChatMessage("assistant", INLINE)
    assert emit_messages(ex)["messages"][1] == {"role": "assistant", "content": INLINE}


def test_flat_record_reasoning_columns_are_opt_in() -> None:
    row = {"question": "q", "trace": "t", "attempt": "a", "reasoning": "on"}
    plain = _one(row, output_keys=("attempt",))
    assert plain.messages[-1] == ChatMessage("assistant", "a")  # "reasoning": "on" is a flag
    ex = _one(row, output_keys=("attempt",), record_reasoning_keys=("trace",))
    assert ex.messages[-1] == ChatMessage("assistant", "a", reasoning="t")


def test_nemotron_input_turns_and_output() -> None:
    ex = _one(
        {
            "input": [{"role": "user", "content": "hi"}],
            "output": INLINE,
            "reasoning": "on",
            "system_prompt": "detailed thinking on",
        }
    )
    assert ex.messages == [
        ChatMessage("system", "detailed thinking on"),
        ChatMessage("user", "hi"),
        ChatMessage("assistant", INLINE),
    ]


def test_reasoning_survives_media_and_hermes_rewrites() -> None:
    ex = _one(
        {
            "messages": [
                {"role": "user", "content": "<image> what?"},
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name": "f", "arguments": {}}</tool_call>',
                    "reasoning_content": "look first",
                },
            ],
            "images": ["a.png"],
            "tools": [{"name": "f", "parameters": {}}],
        }
    )
    assert ex.messages[1].reasoning == "look first"
    assert ex.messages[1].tool_calls[0].name == "f"


# --- writing -------------------------------------------------------------------

R_EX = TrainingExample(
    messages=[
        ChatMessage("user", "q1"),
        ChatMessage("assistant", "a1", reasoning="r1"),
        ChatMessage("user", "q2"),
        ChatMessage("assistant", "<think>\nr2\n</think>\n\na2"),
    ]
)


def _assistants(row: dict) -> list[dict]:
    return [m for m in row["messages"] if m["role"] == "assistant"]


def test_messages_reasoning_modes() -> None:
    keep = _assistants(emit_messages(R_EX))
    assert keep == [
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "assistant", "content": "<think>\nr2\n</think>\n\na2"},
    ]
    for mode in ("reasoning_content", "thinking"):
        rows = _assistants(emit_messages(R_EX, options=EmitOptions(reasoning=mode)))
        assert rows == [
            {"role": "assistant", "content": "a1", mode: "r1"},
            {"role": "assistant", "content": "a2", mode: "r2"},
        ]
    inline = _assistants(emit_messages(R_EX, options=EmitOptions(reasoning="inline")))
    assert [m["content"] for m in inline] == [join_inline("r1", "a1"), join_inline("r2", "a2")]
    drop = _assistants(emit_messages(R_EX, options=EmitOptions(reasoning="drop")))
    assert drop == [{"role": "assistant", "content": "a1"}, {"role": "assistant", "content": "a2"}]


def test_formats_without_a_field_write_reasoning_inline() -> None:
    pair = TrainingExample(messages=R_EX.messages[:2])
    assert emit_alpaca(pair)["output"] == join_inline("r1", "a1")
    assert emit_alpaca(pair, options=EmitOptions(reasoning="drop"))["output"] == "a1"
    sg = emit_sharegpt(R_EX)["conversations"]
    assert [t["value"] for t in sg if t["from"] == "gpt"] == [
        join_inline("r1", "a1"),
        "<think>\nr2\n</think>\n\na2",
    ]


def test_preference_rows_carry_reasoning() -> None:
    ex = TrainingExample(
        messages=[ChatMessage("user", "q"), ChatMessage("assistant", "good", reasoning="r")],
        rejected=[ChatMessage("user", "q"), ChatMessage("assistant", "bad")],
    )
    row = emit_preference(ex, options=EmitOptions(reasoning="thinking"))
    assert row["chosen"] == [{"role": "assistant", "content": "good", "thinking": "r"}]


def test_tool_call_content_is_empty_string_by_default() -> None:
    call = ChatMessage("assistant", None, tool_calls=(ToolCall("f", "{}"),))
    ex = TrainingExample(messages=[ChatMessage("user", "q"), call])
    assert emit_messages(ex)["messages"][1]["content"] == ""
    null = emit_messages(ex, options=EmitOptions(tool_content="null"))
    assert null["messages"][1]["content"] is None


def test_bad_option_values() -> None:
    with pytest.raises(ValueError, match="reasoning must be one of"):
        EmitOptions(reasoning="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tool_content"):
        EmitOptions(tool_content="none")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="has none: use --reasoning inline"):
        build_convert_config(
            adapter="auto", output_format="sharegpt", emit_overrides={"reasoning": "thinking"}
        )


# --- convert, CLI, llamafactory-info ---------------------------------------------


def test_convert_counts_reasoning_and_cli_flags(tmp_path: Path, capsys) -> None:
    src = _write(
        tmp_path / "in.jsonl",
        [
            {"messages": [Q, {"role": "assistant", "content": INLINE}]},
            {"messages": [Q, {"role": "assistant", "content": "a"}]},
        ],
    )
    st = ConvertStats()
    convert_file(src, tmp_path / "o.jsonl", adapter_name="auto", output_format="messages", stats=st)
    assert (st.written, st.reasoning) == (2, 1)
    assert st.to_report()["reasoning"] == 1

    out = tmp_path / "t.jsonl"
    main(["convert", "-i", str(src), "-o", str(out), "--from", "auto", "--format", "messages",
          "--reasoning", "thinking"])  # fmt: skip
    assert _read(out)[0]["messages"][1] == {
        "role": "assistant",
        "content": "The answer is 4.",
        "thinking": "2+2=4",
    }
    assert "1 example carries a reasoning trace" in capsys.readouterr().err

    with pytest.raises(SystemExit) as e:
        main(["convert", "-i", str(src), "-o", str(out), "--from", "auto", "--format",
              "alpaca", "--reasoning", "reasoning_content"])  # fmt: skip
    assert e.value.code == 2
    assert "has none" in capsys.readouterr().err


def test_llamafactory_info_rejects_reasoning_fields(tmp_path: Path) -> None:
    rows = [{"messages": [Q, {"role": "assistant", "content": "a", "reasoning_content": "r"}]}]
    with pytest.raises(ValueError, match="--reasoning inline"):
        dataset_info_entry(_write(tmp_path / "r.jsonl", rows))
    rows[0]["messages"][1] = {"role": "assistant", "content": join_inline("r", "a")}
    assert dataset_info_entry(_write(tmp_path / "i.jsonl", rows))["formatting"] == "sharegpt"
