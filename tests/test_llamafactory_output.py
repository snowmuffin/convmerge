"""--format sharegpt / sharegpt-preference and `convmerge llamafactory-info`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge import ConvertStats, build_convert_config, convert_with_config
from convmerge.cli import main
from convmerge.llamafactory import dataset_info_entry

CATALOG = {e["id"]: e for e in json.loads(
    (Path(__file__).parent / "datasets" / "catalog.json").read_text(encoding="utf-8"))}  # fmt: skip


def _convert(
    tmp_path: Path, records: list[dict], fmt: str, **kw
) -> tuple[list[dict], ConvertStats]:
    src, dst = tmp_path / "in.jsonl", tmp_path / f"{fmt}.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), "utf-8")
    stats = ConvertStats()
    convert_with_config(src, dst, build_convert_config(adapter="auto", output_format=fmt, **kw),
                        stats=stats)  # fmt: skip
    return [json.loads(x) for x in dst.read_text(encoding="utf-8").splitlines()], stats


def M(*turns: tuple[str, str]) -> list[dict]:
    return [{"role": r, "content": c} for r, c in turns]


def test_plain_conversation_with_system(tmp_path: Path) -> None:
    rows, _ = _convert(
        tmp_path,
        [{"messages": M(("system", "Be brief."), ("user", "Hi"), ("assistant", "Hello"),
                        ("user", "Bye"), ("assistant", "Bye!"))}],
        "sharegpt",
    )  # fmt: skip
    assert rows == [
        {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"},
                {"from": "human", "value": "Bye"},
                {"from": "gpt", "value": "Bye!"},
            ],
            "system": "Be brief.",
        }
    ]


def test_tool_calls_become_function_call_and_observation(tmp_path: Path) -> None:
    rows, stats = _convert(tmp_path, [CATALOG["glaiveai/glaive-function-calling-v2"]["record"]],
                           "sharegpt")  # fmt: skip
    row = rows[0]
    assert [t["from"] for t in row["conversations"]] == [
        "human", "function_call", "observation", "gpt",
    ]  # fmt: skip
    assert json.loads(row["conversations"][1]["value"]) == {
        "name": "get_news",
        "arguments": {"country": "Japan"},
    }
    assert json.loads(row["tools"])[0]["name"] == "get_news"  # bare function specs
    assert row["system"].startswith("You are a helpful assistant")
    assert stats.lossy == {}


def test_parallel_calls_text_next_to_calls_and_joined_results(tmp_path: Path) -> None:
    call = lambda n: {"type": "function", "function": {"name": n, "arguments": "{}"}}  # noqa: E731
    record = {"messages": [
        {"role": "user", "content": "Do both"},
        {"role": "assistant", "content": "On it.", "tool_calls": [call("a"), call("b")]},
        {"role": "tool", "content": "A done"},
        {"role": "tool", "content": "B done"},
        {"role": "assistant", "content": "Both done."},
    ]}  # fmt: skip
    rows, stats = _convert(tmp_path, [record], "sharegpt")
    conv = rows[0]["conversations"]
    assert json.loads(conv[1]["value"]) == [
        {"name": "a", "arguments": {}},
        {"name": "b", "arguments": {}},
    ]
    assert conv[2] == {"from": "observation", "value": "A done\nB done"}
    assert stats.lossy == {"lossy_tool_call_text": 1}


def test_media_become_tokens_and_columns(tmp_path: Path) -> None:
    rows, _ = _convert(tmp_path, [CATALOG["liuhaotian/LLaVA-Instruct-150K"]["record"]], "sharegpt")
    assert rows[0]["conversations"][0]["value"] == "<image>\nWhat colors is the bus?"
    assert rows[0]["images"] == ["000000033471.jpg"]


@pytest.mark.parametrize(
    "messages",
    [
        M(("user", "a"), ("user", "b"), ("assistant", "c")),
        M(("user", "a"), ("assistant", "b"), ("user", "c")),
        M(("user", "a"), ("system", "late"), ("assistant", "b")),
    ],
)
def test_turns_that_do_not_alternate_are_dropped(tmp_path: Path, messages: list[dict]) -> None:
    rows, stats = _convert(tmp_path, [{"messages": messages}], "sharegpt")
    assert rows == [] and stats.drop_reasons == {"unrepresentable_role_order": 1}


def test_every_sft_and_tool_catalog_dataset_fits_sharegpt(tmp_path: Path) -> None:
    for e in CATALOG.values():
        if e["kind"] == "preference":
            continue
        _, stats = _convert(tmp_path, [e["record"]], "sharegpt")
        assert stats.written == 1, (e["id"], stats.drop_reasons)


@pytest.mark.parametrize(
    "entry", [e for e in CATALOG.values() if e["kind"] == "preference"], ids=lambda e: e["id"]
)
def test_catalog_preference_datasets_as_ranking(tmp_path: Path, entry: dict) -> None:
    rows, stats = _convert(tmp_path, [entry["record"]], "sharegpt-preference")
    assert stats.written == 1, stats.drop_reasons
    row = rows[0]
    assert row["conversations"][-1]["from"] == "human"
    assert row["chosen"]["from"] == row["rejected"]["from"] == "gpt"
    assert row["chosen"]["value"] != row["rejected"]["value"]


def test_ranking_needs_single_text_answers(tmp_path: Path) -> None:
    call = {"type": "function", "function": {"name": "f", "arguments": "{}"}}
    tool_answer = [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "content": "r"},
        {"role": "assistant", "content": "done"},
    ]
    record = {"prompt": M(("user", "q")), "chosen": tool_answer, "rejected": M(("assistant", "x"))}
    rows, stats = _convert(tmp_path, [record], "sharegpt-preference")
    assert rows == [] and stats.drop_reasons == {"unrepresentable_pair_continuation": 1}
    rows, _ = _convert(tmp_path, [record], "preference")  # the TRL format keeps it
    assert len(rows[0]["chosen"]) == 3


def test_round_trip_through_the_sharegpt_adapter(tmp_path: Path) -> None:
    original = {"messages": M(("system", "S"), ("user", "Hi"), ("assistant", "Hello"))}
    rows, _ = _convert(tmp_path, [original], "sharegpt")
    back, _ = _convert(tmp_path, rows, "messages")
    assert back == [original]


def test_dataset_info_entries(tmp_path: Path) -> None:
    records = [
        CATALOG["glaiveai/glaive-function-calling-v2"]["record"],
        CATALOG["liuhaotian/LLaVA-Instruct-150K"]["record"],
    ]
    _convert(tmp_path, records, "sharegpt")
    entry = dataset_info_entry(tmp_path / "sharegpt.jsonl")
    assert entry == {
        "file_name": "sharegpt.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system", "tools": "tools",
                    "images": "images"},
        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human",
                 "assistant_tag": "gpt", "observation_tag": "observation",
                 "function_tag": "function_call", "system_tag": "system"},
    }  # fmt: skip
    _convert(tmp_path, [CATALOG["Anthropic/hh-rlhf"]["record"]], "sharegpt-preference")
    ranking = dataset_info_entry(tmp_path / "sharegpt-preference.jsonl")
    assert ranking["ranking"] is True
    assert ranking["columns"] == {"messages": "conversations", "chosen": "chosen",
                                  "rejected": "rejected"}  # fmt: skip
    _convert(tmp_path, [{"instruction": "q", "output": "a", "system": "s"}], "alpaca")
    assert dataset_info_entry(tmp_path / "alpaca.jsonl")["columns"] == {
        "prompt": "instruction", "query": "input", "response": "output", "system": "system",
    }  # fmt: skip
    _convert(tmp_path, [{"messages": M(("user", "q"), ("assistant", "a"))}], "messages")
    assert dataset_info_entry(tmp_path / "messages.jsonl")["tags"]["role_tag"] == "role"


@pytest.mark.parametrize(
    ("record", "fmt", "message"),
    [
        (CATALOG["Anthropic/hh-rlhf"]["record"], "preference", "sharegpt-preference"),
        (
            CATALOG["Salesforce/xlam-function-calling-60k"]["record"],
            "messages",
            "--format sharegpt",
        ),
    ],
)
def test_dataset_info_refuses_what_llamafactory_cannot_read(
    tmp_path: Path, record: dict, fmt: str, message: str
) -> None:
    _convert(tmp_path, [record], fmt)
    with pytest.raises(ValueError, match=message):
        dataset_info_entry(tmp_path / f"{fmt}.jsonl")


def test_cli_merges_into_dataset_info(tmp_path: Path, capsys) -> None:
    data = tmp_path / "data"
    data.mkdir()
    rows, _ = _convert(tmp_path, [{"messages": M(("user", "q"), ("assistant", "a"))}], "sharegpt")
    (data / "sft").mkdir()
    (data / "sft" / "train.jsonl").write_text(json.dumps(rows[0]) + "\n")
    info = data / "dataset_info.json"
    info.write_text(json.dumps({"existing": {"file_name": "x.json"}}))
    main(["llamafactory-info", "-i", str(data / "sft" / "train.jsonl"), "--name", "mine",
          "--info", str(info)])  # fmt: skip
    merged = json.loads(info.read_text())
    assert set(merged) == {"existing", "mine"}
    assert merged["mine"]["file_name"] == "sft/train.jsonl"
    assert "'mine' updated" in capsys.readouterr().err
    main(["llamafactory-info", "-i", str(data / "sft" / "train.jsonl"), "--name", "mine"])
    assert json.loads(capsys.readouterr().out)["mine"]["file_name"] == "train.jsonl"


def test_validate_checks_preference_rows(tmp_path: Path, capsys) -> None:
    good, _ = _convert(tmp_path, [CATALOG["Anthropic/hh-rlhf"]["record"]], "preference")
    bad = {"prompt": M(("user", "q")), "chosen": M(("assistant", "same")),
           "rejected": M(("assistant", "same"))}  # fmt: skip
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(good[0]) + "\n")
    main(["validate", "-i", str(path)])
    assert json.loads(capsys.readouterr().out)["valid"] == 1
    path.write_text(json.dumps(good[0]) + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(SystemExit) as exc:
        main(["validate", "-i", str(path)])
    assert exc.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["drop_reasons"] == {"unrepresentable_identical_pair": 1}
