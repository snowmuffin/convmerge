"""--format preference: chosen/rejected pairs in TRL's conversational format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge import ConvertStats, convert_file
from convmerge.adapter_resolve import resolve_adapter
from convmerge.cli import main
from convmerge.config import AdapterOptions
from convmerge.emitters import emit_preference
from convmerge.models import ChatMessage, TrainingExample


def M(*turns: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": r, "content": c} for r, c in turns]


def _convert(tmp_path: Path, records: list[dict], fmt: str = "preference", **kw):
    src = tmp_path / "in.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    stats = ConvertStats()
    convert_file(src, tmp_path / "out.jsonl", adapter_name="auto", output_format=fmt,
                 stats=stats, **kw)  # fmt: skip
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    return rows, stats


SHAPES = {
    "ultrafeedback_binarized": {
        "prompt": "q",
        "prompt_id": "1",
        "chosen": M(("user", "q"), ("assistant", "good")),
        "rejected": M(("user", "q"), ("assistant", "bad")),
        "messages": M(("user", "q"), ("assistant", "good")),
    },
    "hh_rlhf": {
        "chosen": "\n\nHuman: q\n\nAssistant: good",
        "rejected": "\n\nHuman: q\n\nAssistant: bad",
    },
    "orca_dpo_pairs": {"question": "q", "chosen": "good", "rejected": "bad"},
    "llamafactory_ranking": {
        "conversations": [{"from": "human", "value": "q"}],
        "chosen": {"from": "gpt", "value": "good"},
        "rejected": {"from": "gpt", "value": "bad"},
    },
    "trl_implicit_prompt": {
        "chosen": M(("user", "q"), ("assistant", "good")),
        "rejected": M(("user", "q"), ("assistant", "bad")),
    },
    "trl_explicit_prompt": {
        "prompt": M(("user", "q")),
        "chosen": M(("assistant", "good")),
        "rejected": M(("assistant", "bad")),
    },
    "arena_winner": {
        "conversation_a": M(("user", "q"), ("assistant", "bad")),
        "conversation_b": M(("user", "q"), ("assistant", "good")),
        "winner": "model_b",
    },
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_every_preference_shape_becomes_a_pair(tmp_path: Path, shape: str) -> None:
    rows, stats = _convert(tmp_path, [SHAPES[shape]])
    assert stats.dropped == 0
    assert rows == [
        {
            "prompt": [{"role": "user", "content": "q"}],
            "chosen": [{"role": "assistant", "content": "good"}],
            "rejected": [{"role": "assistant", "content": "bad"}],
        }
    ]


def test_shared_multi_turn_prefix_and_system_become_the_prompt(tmp_path: Path) -> None:
    record = {
        "system": "be nice",
        "chosen": "\n\nHuman: a\n\nAssistant: b\n\nHuman: c\n\nAssistant: good",
        "rejected": "\n\nHuman: a\n\nAssistant: b\n\nHuman: c\n\nAssistant: bad",
    }
    rows, _ = _convert(tmp_path, [record])
    assert [m["content"] for m in rows[0]["prompt"]] == ["be nice", "a", "b", "c"]
    assert rows[0]["chosen"] == [{"role": "assistant", "content": "good"}]


def test_unpaired_identical_and_incomplete_are_dropped_with_reasons(tmp_path: Path) -> None:
    qx = M(("user", "q"), ("assistant", "x"))
    records = [
        {"instruction": "q", "output": "a"},
        {"chosen": qx, "rejected": qx},
        {"chosen": qx, "rejected": M(("user", "q"))},
        # An arena tie is neither a pair nor (with the default winner mode) an example.
        {"conversation_a": qx, "conversation_b": qx, "winner": "tie"},
    ]
    rows, stats = _convert(tmp_path, records)
    assert rows == [] and stats.no_example == 1
    assert stats.drop_reasons == {
        "unrepresentable_not_preference": 1,
        "unrepresentable_identical_pair": 1,
        "unrepresentable_incomplete_pair": 1,
    }
    report = stats.to_report()["reason_descriptions"]
    assert "identical" in report["unrepresentable_identical_pair"]


def test_tools_meta_and_object_arguments_carry_over(tmp_path: Path) -> None:
    from convmerge import EmitOptions

    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}
    answer = {"role": "assistant", "content": None, "tool_calls": [call]}
    record = {
        "id": "r1",
        "tools": [{"name": "f", "parameters": {}}],
        "chosen": [{"role": "user", "content": "q"}, answer],
        "rejected": M(("user", "q"), ("assistant", "no")),
    }
    rows, _ = _convert(
        tmp_path, [record], emit_options=EmitOptions(keep_meta=["id"], tool_arguments="object")
    )
    row = rows[0]
    assert row["tools"] == [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    assert row["chosen"][0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert row["meta"] == {"id": "r1"}


def test_sft_conversion_explains_preference_records(tmp_path: Path, capsys) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(
        json.dumps(SHAPES["hh_rlhf"]) + "\n" + json.dumps(SHAPES["orca_dpo_pairs"]) + "\n"
    )
    main(["convert", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--from", "auto",
          "--format", "messages"])  # fmt: skip
    err = capsys.readouterr().err
    assert "preference_record=2" in err
    assert "hint: preference_record" in err and "--format preference" in err


def test_valid_sft_view_of_a_preference_record_is_kept(tmp_path: Path) -> None:
    # UltraFeedback-binarized also ships a ``messages`` column: SFT uses it as before.
    rows, stats = _convert(tmp_path, [SHAPES["ultrafeedback_binarized"]], fmt="messages")
    assert stats.dropped == 0 and rows[0]["messages"][1]["content"] == "good"


def test_preference_option_conflicts_with_pairs(tmp_path: Path, capsys) -> None:
    with pytest.raises(ValueError, match="folds pairs"):
        resolve_adapter("auto", AdapterOptions(preference="chosen"), pairs=True)
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(SHAPES["hh_rlhf"]) + "\n")
    with pytest.raises(SystemExit) as exc:
        main(["convert", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--from", "auto",
              "--format", "preference", "--preference", "chosen"])  # fmt: skip
    assert exc.value.code == 2
    assert "use one or the other" in capsys.readouterr().err


def test_parallel_matches_single_process(tmp_path: Path) -> None:
    records = [SHAPES[s] for s in sorted(SHAPES)] * 700
    single, s1 = _convert(tmp_path, records)
    multi, s2 = _convert(tmp_path, records, workers=3)
    assert single == multi and s1 == s2 and len(single) == len(records)


def test_emitter_without_rejected_raises() -> None:
    from convmerge import UnrepresentableExample

    ex = TrainingExample([ChatMessage("user", "q"), ChatMessage("assistant", "a")])
    with pytest.raises(UnrepresentableExample) as exc:
        emit_preference(ex)
    assert exc.value.reason == "unrepresentable_not_preference"
