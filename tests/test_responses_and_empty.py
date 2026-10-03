"""A question with a list of model responses, and empty answers (1.6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.adapters.chat import iter_from_chat_line
from convmerge.cli import main
from convmerge.validate import validate_example

NATURAL_REASONING = {
    "question": "What is the work done lifting a mass m by h?",
    "reference_answer": "W = mgh",
    "responses": [{"response_model": "Llama-3.3-70B-Instruct",
                   "response": "## Step 1\nThe work equals the change in potential energy: mgh."}],
}  # fmt: skip


def _one(record):
    return list(iter_from_chat_line(record))


def test_question_with_responses_takes_the_first_response() -> None:
    (ex,) = _one(NATURAL_REASONING)
    assert not validate_example(ex)
    assert [(m.role, m.content) for m in ex.messages] == [
        ("user", NATURAL_REASONING["question"]),
        ("assistant", NATURAL_REASONING["responses"][0]["response"]),
    ]
    two = {**NATURAL_REASONING, "responses": [{"response": "  "}, {"response": "second"}]}
    assert _one(two)[0].messages[-1].content == "second"


@pytest.mark.parametrize(
    "responses", [[], [{"response": " "}], ["plain string"], "not a list", [{"text": "x"}]]
)
def test_responses_without_a_usable_answer_convert_nothing(responses) -> None:
    examples = _one({**NATURAL_REASONING, "responses": responses})
    assert all(validate_example(ex) for ex in examples)


@pytest.mark.parametrize(
    "record",
    [
        {"instruction": "Name a color", "input": "", "output": "",
         "text": "### Instruction:\\nName a color\\n\\n### Response:\\n"},
        {"instruction": "Name a color", "output": "   "},
        {"question": "q", "answer": ""},
    ],
)  # fmt: skip
def test_empty_answers_are_dropped_as_empty_answer(record) -> None:
    (ex,) = _one(record)
    assert ex.issues == ["empty_answer"] and not ex.messages


def test_an_empty_column_beside_a_filled_one_is_not_empty() -> None:
    (ex,) = _one({"instruction": "q", "output": "", "response": "an answer"})
    assert not validate_example(ex) and ex.messages[-1].content == "an answer"


def test_convert_reports_empty_answer_without_the_mapping_hint(tmp_path: Path, capsys) -> None:
    src = tmp_path / "a.jsonl"
    row = {"instruction": "q", "output": "", "text": "### Instruction:"}
    src.write_text(json.dumps(row) + "\n")
    main(["convert", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--from", "auto"])
    err = capsys.readouterr().err
    assert "empty_answer=1" in err
    assert "--from map" not in err and "routing record to the 'text' branch" not in err
