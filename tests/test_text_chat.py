"""Template-rendered ``text`` columns are split back into turns."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge import ConvertStats, convert_file
from convmerge.adapters.text_chat import parse_text_chat


def _roles(text: str) -> list[tuple[str, str]]:
    turns = parse_text_chat(text)
    assert turns is not None
    return [(m.role, m.text) for m in turns]


def test_chatml_oasst_top1() -> None:
    text = (
        "<|im_start|>user\nExplain calculus<|im_end|>\n"
        "<|im_start|>assistant\nIt studies change.\n\nTwo parts.<|im_end|>\n"
        "<|im_start|>user\nThanks<|im_end|>\n<|im_start|>assistant\nAnytime.<|im_end|>\n"
    )
    assert _roles(text) == [
        ("user", "Explain calculus"),
        ("assistant", "It studies change.\n\nTwo parts."),
        ("user", "Thanks"),
        ("assistant", "Anytime."),
    ]


def test_guanaco_drops_the_unanswered_last_question() -> None:
    text = (
        "### Human: What is monopsony?### Assistant: One buyer.### Human: Now explain it to a dog"
    )
    assert _roles(text) == [("user", "What is monopsony?"), ("assistant", "One buyer.")]


def test_hh_transcript() -> None:
    text = "\n\nHuman: Hi\n\nAssistant: Hello\n\nHuman: Joke?\n\nAssistant: Knock knock."
    assert _roles(text) == [
        ("user", "Hi"),
        ("assistant", "Hello"),
        ("user", "Joke?"),
        ("assistant", "Knock knock."),
    ]


def test_llama2_with_system_and_two_turns() -> None:
    text = (
        "<s>[INST] <<SYS>>\nBe brief.\n<</SYS>>\n\nHi [/INST] Hello </s>"
        "<s>[INST] Bye [/INST] Goodbye </s>"
    )
    assert _roles(text) == [
        ("system", "Be brief."),
        ("user", "Hi"),
        ("assistant", "Hello"),
        ("user", "Bye"),
        ("assistant", "Goodbye"),
    ]


def test_llama3() -> None:
    text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe kind.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>"
    )
    assert _roles(text) == [("system", "Be kind."), ("user", "Hi"), ("assistant", "Hello!")]


def test_gemma() -> None:
    text = "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n"
    assert _roles(text) == [("user", "Hi"), ("assistant", "Hello!")]


def test_alpaca_prompt_template() -> None:
    text = (
        "Below is an instruction that describes a task, paired with an input that provides "
        "further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nTranslate to French\n\n### Input:\ncat\n\n### Response:\nchat"
    )
    assert _roles(text) == [("user", "Translate to French\ncat"), ("assistant", "chat")]


@pytest.mark.parametrize("text", ["just some prose", "Human rights matter.", "### Notes: x"])
def test_unrecognized_text_is_not_parsed(text: str) -> None:
    assert parse_text_chat(text) is None


def _convert(tmp_path: Path, records: list[dict]) -> tuple[list[dict], ConvertStats]:
    src = tmp_path / "in.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    stats = ConvertStats()
    convert_file(src, tmp_path / "o.jsonl", adapter_name="auto", output_format="messages",
                 stats=stats)  # fmt: skip
    return [json.loads(x) for x in (tmp_path / "o.jsonl").read_text().splitlines()], stats


def test_auto_adapter_reads_text_columns(tmp_path: Path) -> None:
    rows, stats = _convert(
        tmp_path,
        [
            {"text": "### Human: Q### Assistant: A"},
            {"text": "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nA<|im_end|>"},
            {"text": "plain prose, not a chat"},  # still dropped, as before
            # Alpaca keys still win over a text column (issue #17).
            {"instruction": "Q", "output": "A", "text": "### Human: other### Assistant: other"},
        ],
    )
    assert [r["messages"] for r in rows] == [
        [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    ] * 3
    assert stats.drop_reasons == {"no_user": 1}


def test_system_column_joins_parsed_text(tmp_path: Path) -> None:
    rows, _ = _convert(tmp_path, [{"system": "S", "text": "### Human: Q### Assistant: A"}])
    assert rows[0]["messages"][0] == {"role": "system", "content": "S"}
