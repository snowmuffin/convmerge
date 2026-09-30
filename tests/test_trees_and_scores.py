"""OpenAssistant message trees and scored candidates (1.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge import ConvertStats, convert_file, convert_records
from convmerge.adapter_resolve import resolve_adapter
from convmerge.adapters.mapped import MapSpec
from convmerge.config import AdapterOptions
from convmerge.validate import validate_example

SFT = resolve_adapter("auto", None)
PAIRS = resolve_adapter("auto", None, pairs=True)
CHOSEN = resolve_adapter("auto", AdapterOptions(preference="chosen"))


def _msg(mid, parent, role, text, rank=None, tree="t1", **extra):
    return {"message_id": mid, "parent_id": parent, "role": role, "text": text, "rank": rank,
            "message_tree_id": tree, "deleted": False, "review_result": True, **extra}  # fmt: skip


# prompt → two ranked answers; the best one has a follow-up and an answer.
FLAT = [
    _msg("a", None, "prompter", "Why can't we divide by 0?"),
    _msg("b", "a", "assistant", "Because it is undefined.", rank=1),
    _msg("c", "a", "assistant", "Division has no inverse for 0.", rank=0),
    _msg("d", "c", "prompter", "Explain more?"),
    _msg("e", "d", "assistant", "No number times 0 gives 1.", rank=0),
    _msg("f", "a", "assistant", "spam", rank=None, deleted=True),
]


def _nest(rows, mid):
    node = next(r for r in rows if r["message_id"] == mid)
    kids = [_nest(rows, r["message_id"]) for r in rows if r["parent_id"] == mid]
    return {k: v for k, v in node.items() if k not in ("parent_id", "message_tree_id")} | {
        "replies": kids
    }


TREE = {"message_tree_id": "t1", "tree_state": "ready_for_export", "prompt": _nest(FLAT, "a")}
BEST = [
    ("user", "Why can't we divide by 0?"),
    ("assistant", "Division has no inverse for 0."),
    ("user", "Explain more?"),
    ("assistant", "No number times 0 gives 1."),
]


def _turns(messages):
    return [(m["role"], m["content"]) for m in messages]


def _write(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_tree_row_follows_the_best_ranked_reply() -> None:
    [ex] = list(SFT(TREE))
    assert [(m.role, m.text) for m in ex.messages] == BEST
    assert ex.meta["source"] == "chat:oasst"


def test_tree_ending_on_a_prompt_drops_the_unanswered_turn() -> None:
    rows = [r for r in FLAT if r["message_id"] != "e"]
    tree = {"message_tree_id": "t1", "prompt": _nest(rows, "a")}
    [ex] = list(SFT(tree))
    assert [(m.role, m.text) for m in ex.messages] == BEST[:2]


def test_flat_message_rows_become_one_conversation_per_tree(tmp_path: Path) -> None:
    other = [
        _msg("x", None, "prompter", "Hola", tree="t2"),
        _msg("y", "x", "assistant", "¡Hola!", rank=0, tree="t2"),
    ]
    src = _write(tmp_path / "in.jsonl", [*FLAT, *other])
    stats = ConvertStats()
    convert_file(src, tmp_path / "out.jsonl", adapter_name="auto", output_format="messages",
                 stats=stats)  # fmt: skip
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    hola = [("user", "Hola"), ("assistant", "¡Hola!")]
    assert [_turns(r["messages"]) for r in rows] == [BEST, hola]
    assert (stats.lines_read, stats.grouped, stats.written, stats.skipped) == (8, 6, 2, 0)
    # the in-memory API groups the same way, and --workers falls back to one process
    in_memory = convert_records([*FLAT, *other])
    assert [r["messages"] for r in in_memory] == [r["messages"] for r in rows]
    convert_file(src, tmp_path / "w.jsonl", adapter_name="auto", output_format="messages",
                 workers=2)  # fmt: skip
    assert (tmp_path / "w.jsonl").read_text() == (tmp_path / "out.jsonl").read_text()


def test_tree_without_its_root_is_reported(tmp_path: Path, capsys) -> None:
    src = _write(tmp_path / "in.jsonl", FLAT[1:])  # the file starts mid-tree
    stats = ConvertStats()
    convert_file(src, tmp_path / "out.jsonl", adapter_name="auto", output_format="messages",
                 stats=stats)  # fmt: skip
    assert stats.drop_reasons == {"missing_root": 1}


def test_tree_pairs_the_best_and_worst_ranked_answers() -> None:
    [ex] = list(PAIRS(TREE))
    # The follow-up has one answer, so the pair comes from the first step.
    assert [(m.role, m.text) for m in ex.messages] == BEST[:2]
    worst = ("assistant", "Because it is undefined.")
    assert [(m.role, m.text) for m in ex.rejected] == [BEST[0], worst]


def test_tree_pair_uses_the_deepest_ranked_step() -> None:
    rows = [*FLAT, _msg("g", "d", "assistant", "It just is.", rank=1)]
    [ex] = list(PAIRS({"message_tree_id": "t1", "prompt": _nest(rows, "a")}))
    assert [(m.role, m.text) for m in ex.messages] == BEST
    assert ex.rejected[-1].text == "It just is."
    assert len(ex.rejected) == 4


def test_tree_without_ranked_siblings_is_not_a_pair(tmp_path: Path) -> None:
    rows = [r for r in FLAT if r["message_id"] != "b"]
    out = list(convert_records(rows, output_format="preference"))
    assert out == []


ULTRAFEEDBACK = {
    "source": "evol_instruct",
    "instruction": "Name a prime.",
    "models": ["m1", "m2", "m3"],
    "completions": [
        {"model": "m1", "response": "4", "fine-grained_score": 1.25, "overall_score": 9.0},
        {"model": "m2", "response": "7", "fine-grained_score": 4.75, "overall_score": 7.5},
        {"model": "m3", "response": "Maybe 9", "fine-grained_score": 2.5, "overall_score": 6.0},
    ],
}
NECTAR = {
    "prompt": "\n\nHuman: What is 2+2?\n\nAssistant: ",
    "answers": [
        {"answer": "5", "model": "a", "rank": 2.0},
        {"answer": "4", "model": "b", "rank": 1.0},
        {"answer": "22", "model": "c", "rank": 3.0},
    ],
    "turns": 1,
    "num_responses": 3,
    "good_natured": True,
}


def _pair(ex):
    return [(m.role, m.text) for m in ex.messages], [(m.role, m.text) for m in ex.rejected]


def test_ultrafeedback_pairs_by_fine_grained_score() -> None:
    [ex] = list(PAIRS(ULTRAFEEDBACK))
    assert _pair(ex) == (
        [("user", "Name a prime."), ("assistant", "7")],
        [("user", "Name a prime."), ("assistant", "4")],
    )
    [sft] = list(CHOSEN(ULTRAFEEDBACK))
    assert [m.text for m in sft.messages] == ["Name a prime.", "7"]


def test_ultrafeedback_falls_back_to_overall_score() -> None:
    completions = [
        {k: v for k, v in c.items() if k != "fine-grained_score"}
        for c in ULTRAFEEDBACK["completions"]
    ]
    record = {**ULTRAFEEDBACK, "completions": completions}
    [ex] = list(PAIRS(record))
    assert ex.messages[-1].text == "4" and ex.rejected[-1].text == "Maybe 9"


def test_nectar_pairs_rank_one_with_the_last_rank() -> None:
    [ex] = list(PAIRS(NECTAR))
    assert _pair(ex) == (
        [("user", "What is 2+2?"), ("assistant", "4")],
        [("user", "What is 2+2?"), ("assistant", "22")],
    )


@pytest.mark.parametrize("record", [ULTRAFEEDBACK, NECTAR])
def test_scored_rows_without_a_flag_say_how_to_read_them(record) -> None:
    [ex] = list(SFT(record))
    assert "preference_record" in validate_example(ex)


def test_tied_scores_are_no_preference() -> None:
    tied = {**NECTAR, "answers": [{"answer": "4", "rank": 1}, {"answer": "four", "rank": 1}]}
    for adapter in (PAIRS, CHOSEN):
        [ex] = list(adapter(tied))
        assert validate_example(ex) == ["no_preference"]


def test_map_scored_candidates() -> None:
    spec = MapSpec.from_mapping({
        "user": "instruction", "candidates": "completions", "candidate": "response",
        "score": "overall_score",
    })  # fmt: skip
    pairs = resolve_adapter("map", AdapterOptions(map=spec), pairs=True)
    [ex] = list(pairs(ULTRAFEEDBACK))
    assert (ex.messages[-1].text, ex.rejected[-1].text) == ("4", "Maybe 9")
    ranks = MapSpec.from_mapping({
        "user": "q", "candidates": "a[]", "candidate": "t", "score": "r", "better": "lower",
    })  # fmt: skip
    record = {"q": "Hi", "a": [{"t": "x", "r": 2}, {"t": "y", "r": "1"}, {"t": "", "r": 0}]}
    [ex] = list(resolve_adapter("map", AdapterOptions(map=ranks), pairs=True)(record))
    assert (ex.messages[-1].text, ex.rejected[-1].text) == ("y", "x")
    assert ranks.to_mapping()["better"] == "lower"
    [tie] = list(resolve_adapter("map", AdapterOptions(map=ranks), pairs=True)(
        {"q": "Hi", "a": [{"t": "x", "r": 1}]}
    ))  # fmt: skip
    assert validate_example(tie) == ["no_preference"]


@pytest.mark.parametrize(
    "mapping, message",
    [
        ({"user": "q", "candidates": "a"}, "go together"),
        ({"user": "q", "candidates": "a", "candidate": "t", "score": "s", "better": "up"},
         "higher"),
        ({"user": "q", "assistant": "x", "better": "lower"}, "needs"),
        ({"user": "q", "chosen": "c", "rejected": "r", "candidates": "a", "candidate": "t",
          "score": "s"}, "one of"),
    ],
)  # fmt: skip
def test_map_scored_candidates_errors(mapping, message) -> None:
    with pytest.raises(ValueError, match=message):
        MapSpec.from_mapping(mapping)


def _flags(rows):
    return [[m.get("train") for m in r["messages"] if m["role"] == "assistant"] for r in rows]


def test_dataset_turn_flags_with_train_turns_data(tmp_path: Path) -> None:
    from convmerge import EmitOptions, TransformOptions
    from convmerge.axolotl import dataset_config

    records = [
        {"messages": [  # a per-turn "loss" mask
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "loss": False},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "loss": True},
        ]},
        {"conversations": [  # axolotl ShareGPT: "weight" 0 / 1
            {"from": "human", "value": "q"}, {"from": "gpt", "value": "a", "weight": 0},
            {"from": "human", "value": "q"}, {"from": "gpt", "value": "b"},
        ]},
        {"messages": [  # Nemotron: one flag per turn in metadata.train_turns
            {"role": "system", "content": None}, {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ], "metadata": {"train_turns": [False, False, False, False, True]}},
    ]  # fmt: skip
    rows = list(convert_records(records, emit_options=EmitOptions(train_turns="data")))
    assert _flags(rows) == [[False, True], [False, True], [False, True]]
    # without the option nothing changes
    assert _flags(list(convert_records(records))) == [[None, None]] * 3
    # merged answers keep a flag when either side is trained
    merged = list(convert_records(
        [{"messages": [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": "a", "train": False},
                       {"role": "assistant", "content": "b", "train": True}]}],
        emit_options=EmitOptions(train_turns="data"),
        transform_options=TransformOptions(merge_consecutive=True),
    ))  # fmt: skip
    assert _flags(merged) == [[True]]
    out = _write(tmp_path / "out.jsonl", rows)
    assert dataset_config(out)["message_field_training"] == "train"


def test_train_turns_data_needs_the_messages_format() -> None:
    from convmerge import build_convert_config

    with pytest.raises(ValueError, match="train_turns: data"):
        build_convert_config(adapter="auto", output_format="alpaca",
                             emit_overrides={"train_turns": "data"})  # fmt: skip


def test_toolbench_react_turns_become_tool_calls() -> None:
    record = {"messages": [
        {"role": "system", "content": "You are AutoGPT, you can use many tools."},
        {"role": "user", "content": "\nOrderbook for XYZ?\nBegin!\n"},
        {"role": "assistant", "content": "\nThought: Look it up.\nAction: get_orderbook\n"
                                         "Action Input: {\n  \"ticker\": \"XYZ\"\n}"},
        {"role": "tool", "content": {"error": "", "response": "{'bids': []}"}},
        {"role": "assistant", "content": "\nThought: Done.\nAction: Finish\nAction Input: "
                                         '{"return_type": "give_answer", "final_answer": "E"}'},
    ]}  # fmt: skip
    [row] = convert_records([record])
    call, tool, finish = row["messages"][2:]
    assert call["content"] == "Look it up."
    assert call["tool_calls"][0]["function"] == {"name": "get_orderbook",
                                                 "arguments": '{"ticker": "XYZ"}'}  # fmt: skip
    assert tool == {"role": "tool", "name": "get_orderbook",
                    "content": '{"error": "", "response": "{\'bids\': []}"}'}  # fmt: skip
    assert finish["tool_calls"][0]["function"]["name"] == "Finish"


def test_react_text_without_a_tool_answer_stays_text() -> None:
    text = 'Use this format:\nAction: search\nAction Input: {"q": 1}'
    record = {"messages": [{"role": "user", "content": "How?"},
                           {"role": "assistant", "content": text}]}  # fmt: skip
    [row] = convert_records([record])
    assert row["messages"][1] == {"role": "assistant", "content": text}


def test_answer_turns_with_the_question_in_a_column() -> None:
    record = {
        "original_question": "Which year?",
        "messages": [
            {"role": "assistant", "content": "<tool_call>{'name': 'web_search', "
                                             "'arguments': {'query': 'year'}}</tool_call>"},
            {"role": "tool", "content": "Observation: 1960"},
            {"role": "assistant", "content": "<tool_call>{'name': 'final_answer', "
                                             "'arguments': {'answer': '1960'}}</tool_call>"},
        ],
    }  # fmt: skip
    [row] = convert_records([record])
    assert [m["role"] for m in row["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert row["messages"][0]["content"] == "Which year?"
    assert row["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"query": "year"}'


def test_odd_message_rows_are_not_trees() -> None:
    rows = [
        {"message_id": {"x": 1}, "parent_id": None, "message_tree_id": "t", "role": "prompter"},
        {"message_id": "a", "parent_id": None, "message_tree_id": "t", "role": {"r": 1}},
        {"message_tree_id": "t", "prompt": {"role": ["x"], "text": "hi", "replies": []}},
    ]
    stats = ConvertStats()
    assert list(convert_records(rows, stats=stats)) == []
    assert stats.grouped == 0
