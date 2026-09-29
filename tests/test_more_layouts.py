"""Layouts recognised since 1.2 (and the rules that keep older records as they were)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge import TransformOptions, convert_file
from convmerge.adapter_resolve import resolve_adapter
from convmerge.validate import validate_example

SFT = resolve_adapter("auto", None)
PAIRS = resolve_adapter("auto", None, pairs=True)


def _one(adapter, record):
    [example] = list(adapter(record))
    return example


def _turns(example):
    return [(m.role, m.text, [c.name for c in m.tool_calls]) for m in example.messages]


def test_inputs_targets() -> None:
    ex = _one(SFT, {"inputs": "Capital of France?", "targets": "Paris.", "language": "English"})
    assert _turns(ex) == [("user", "Capital of France?", []), ("assistant", "Paris.", [])]


def test_known_keys_win_over_inputs_targets() -> None:
    ex = _one(SFT, {"instruction": "q", "output": "a", "inputs": "x", "targets": "y"})
    assert _turns(ex) == [("user", "q", []), ("assistant", "a", [])]


def test_capitalised_keys() -> None:
    ex = _one(SFT, {"Instruction": "질문", "Response": "답변", "Source": "kin"})
    assert _turns(ex) == [("user", "질문", []), ("assistant", "답변", [])]
    # Records with a known key are read exactly as before: "Response" is not an answer here.
    ex = _one(SFT, {"messages": [{"role": "user", "content": "q"}], "Response": "a"})
    assert validate_example(ex) == ["no_assistant"]


def test_input_role_is_the_system_prompt() -> None:
    turns = [("input", "Be brief."), ("human", "Hi"), ("bot", "Hello")]
    record = {"conversations": [{"from": f, "value": v} for f, v in turns]}
    assert [t[0] for t in _turns(_one(SFT, record))] == ["system", "user", "assistant"]


def test_json_columns_and_separate_answer() -> None:
    record = {
        "messages_json": json.dumps([{"role": "user", "content": "Variance of 1, 2, 3?"}]),
        "tools_json": json.dumps([{"name": "variance", "parameters": {"type": "object"}}]),
        "target_json": json.dumps({"tool_calls": [{"name": "variance", "arguments": {"x": [1]}}]}),
    }
    ex = _one(SFT, record)
    assert _turns(ex) == [("user", "Variance of 1, 2, 3?", []), ("assistant", "", ["variance"])]
    assert ex.tools and ex.tools[0]["function"]["name"] == "variance"
    assert validate_example(ex) == []


def test_answer_column_only_fills_a_missing_answer() -> None:
    record = {"messages": [{"role": "user", "content": "q"}], "response": "a"}
    assert _turns(_one(SFT, record))[-1] == ("assistant", "a", [])
    record = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
              "response": "other"}  # fmt: skip
    assert _turns(_one(SFT, record)) == [("user", "q", []), ("assistant", "a", [])]


def test_rendered_conversation_under_a_conversation_key() -> None:
    text = ("<bos><start_of_turn>user\nHi<end_of_turn>\n"
            "<start_of_turn>model\nHello<end_of_turn>\n")  # fmt: skip
    assert _turns(_one(SFT, {"conversation": text})) == [
        ("user", "Hi", []),
        ("assistant", "Hello", []),
    ]


@pytest.mark.parametrize(
    ("chosen", "rejected"),
    [("chosen_response", "rejected_response"), ("response_chosen", "response_rejected")],
)
def test_preference_key_aliases(chosen: str, rejected: str) -> None:
    ex = _one(PAIRS, {"instruction": "2+2?", chosen: "4", rejected: "5"})
    assert [m.text for m in ex.messages] == ["2+2?", "4"]
    assert [m.text for m in ex.rejected] == ["2+2?", "5"]


def test_hh_transcript_without_leading_blank_line() -> None:
    record = {"chosen": "Human: Hi there\n\nAssistant: Hello!",
              "rejected": "Human: Hi there\n\nAssistant: Go away."}  # fmt: skip
    ex = _one(PAIRS, record)
    assert [(m.role, m.text) for m in ex.messages] == [
        ("user", "Hi there"),
        ("assistant", "Hello!"),
    ]
    assert ex.rejected[-1].text == "Go away."
    # A plain answer that happens to start with "Human" is not a transcript.
    ex = _one(PAIRS, {"prompt": "Q", "chosen": "Human rights matter.", "rejected": "No."})
    assert ex.messages[-1].text == "Human rights matter."


def test_glaive_ai_to_call() -> None:
    record = {
        "system": 'SYSTEM: You have functions -\n{"name": "bmi", "parameters": {"type": "object"}}',
        "chat": "USER: My BMI? 1.75 m\n\n\nASSISTANT: Let me check.\n"
        'AI to=bmi: {"height": 1.75} <|endoftext|>\n\n\n'
        'FUNCTION RESPONSE: {"bmi": 22.2}\n\n\nASSISTANT: It is 22.2. <|endoftext|>',
    }
    ex = _one(SFT, record)
    assert _turns(ex)[1:] == [
        ("user", "My BMI? 1.75 m", []),
        ("assistant", "Let me check.", ["bmi"]),
        ("tool", '{"bmi": 22.2}', []),
        ("assistant", "It is 22.2.", []),
    ]
    assert ex.messages[3].name == "bmi" and validate_example(ex) == []


def test_ai_to_needs_a_json_object() -> None:
    answer = {"role": "assistant", "content": "Say AI to=someone: hello"}
    record = {"messages": [{"role": "user", "content": "q"}, answer]}
    assert _turns(_one(SFT, record))[-1] == ("assistant", "Say AI to=someone: hello", [])


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_leading_assistant(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"messages": [
        {"role": "system", "content": None}, {"role": "user", "content": None},
        {"role": "assistant", "content": "a0"}, {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"}]}) + "\n", encoding="utf-8")  # fmt: skip
    out = tmp_path / "out.jsonl"
    assert convert_file(src, out, adapter_name="auto", output_format="messages") == (1, 0)
    convert_file(src, out, adapter_name="auto", output_format="messages",
                 transform_options=TransformOptions(leading_assistant="drop"))  # fmt: skip
    assert _rows(out) == [{"messages": [{"role": "user", "content": "q1"},
                                        {"role": "assistant", "content": "a1"}]}]  # fmt: skip


def test_leading_assistant_cli_and_recipe_keys(tmp_path: Path, capsys) -> None:
    from convmerge.cli import main
    from convmerge.config import transform_options_from_mapping

    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"messages": [{"role": "assistant", "content": "a0"},
                                            {"role": "user", "content": "q"},
                                            {"role": "assistant", "content": "a"}]}) + "\n",
                   encoding="utf-8")  # fmt: skip
    main(["convert", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--from", "auto"])
    # An agent that greets first is a whole conversation, not a withheld prompt.
    assert "wrote 1 examples" in capsys.readouterr().err
    main(["convert", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--from", "auto",
          "--leading-assistant", "drop"])  # fmt: skip
    assert "leading_assistant_dropped=1" in capsys.readouterr().err
    assert transform_options_from_mapping({"leading_assistant": "drop"}).leading_assistant == "drop"
    with pytest.raises(ValueError):
        TransformOptions(leading_assistant="first")


def test_train_turns_last(tmp_path: Path) -> None:
    from convmerge import EmitOptions
    from convmerge.axolotl import dataset_config

    src = tmp_path / "in.jsonl"
    turns = [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]
    src.write_text(json.dumps({"messages": [{"role": r, "content": c} for r, c in turns]}) + "\n",
                   encoding="utf-8")  # fmt: skip
    out = tmp_path / "out.jsonl"
    convert_file(src, out, adapter_name="auto", output_format="messages",
                 emit_options=EmitOptions(train_turns="last"))  # fmt: skip
    msgs = _rows(out)[0]["messages"]
    assert [m.get("train") for m in msgs] == [None, False, None, None]
    assert dataset_config(out)["message_field_training"] == "train"
    convert_file(src, out, adapter_name="auto", output_format="messages")
    assert "message_field_training" not in dataset_config(out)


def test_train_turns_last_needs_the_messages_format() -> None:
    from convmerge import build_convert_config

    with pytest.raises(ValueError, match="mask_history"):
        build_convert_config(adapter="auto", output_format="sharegpt",
                             emit_overrides={"train_turns": "last"})  # fmt: skip


def test_convert_records_matches_convert_file(tmp_path: Path) -> None:
    from convmerge import ConvertStats, InvalidExampleError, convert_records

    records = [
        {"instruction": "Hi", "output": "Hello"},
        {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]},
        "not a record",
        {"messages": [{"role": "user", "content": "no answer"}]},
    ]
    stats = ConvertStats()
    rows = list(convert_records(iter(records), stats=stats))
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    file_stats = ConvertStats()
    convert_file(src, out, adapter_name="auto", output_format="messages", stats=file_stats)
    assert rows == _rows(out)
    assert (stats.lines_read, stats.written, stats.non_object, stats.drop_reasons) == (
        4, 2, 1, {"no_assistant": 1},
    )  # fmt: skip
    assert stats.drop_reasons == file_stats.drop_reasons
    with pytest.raises(InvalidExampleError) as exc:
        list(convert_records(records, on_invalid="fail"))
    assert exc.value.line_number == 4


def test_validate_reports_fields_that_change_type(tmp_path: Path, capsys) -> None:
    from convmerge.arrow import type_conflicts
    from convmerge.cli import main

    call = {"type": "function", "function": {"name": "f", "arguments": {"data": [1]}}}
    rows = [
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "", "tool_calls": [call]}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]},
                      {"role": "assistant", "content": "a", "n": 1.5}]},
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": None,
                                                         "n": 2}]},
    ]  # fmt: skip
    rows[1]["messages"][1]["tool_calls"] = [
        {"type": "function", "function": {"name": "f", "arguments": {"data": "[1]"}}}
    ]
    src = tmp_path / "in.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    found = {c.path: c.first_line for c in type_conflicts(src)}
    # int/float and null do not conflict
    assert found == {
        "messages[].content": {"list": 2, "string": 1},
        "messages[].tool_calls[].function.arguments.data": {"list": 1, "string": 2},
    }
    with pytest.raises(SystemExit):
        main(["validate", "-i", str(src)])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert [c["path"] for c in report["type_conflicts"]] == sorted(found)
    assert "--tool-arguments string" in captured.err


def test_withheld_prompt_needs_a_null_user_turn(tmp_path: Path, capsys) -> None:
    from convmerge.cli import main

    withheld = {"messages": [{"role": "user", "content": None},
                             {"role": "assistant", "content": "a0"},
                             {"role": "user", "content": "q"},
                             {"role": "assistant", "content": "a"}]}  # fmt: skip
    ex = _one(SFT, withheld)
    assert validate_example(ex) == ["withheld_prompt"]
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(withheld) + "\n", encoding="utf-8")
    main(["convert", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--from", "auto"])
    assert "--leading-assistant drop" in capsys.readouterr().err
    # An empty (not null) user turn is a blank turn, skipped as before.
    blank = {"messages": [{"role": "user", "content": ""}, *withheld["messages"][1:]]}
    assert validate_example(_one(SFT, blank)) == []
