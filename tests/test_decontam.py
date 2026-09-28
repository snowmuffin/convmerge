"""convmerge decontam: n-gram overlap with evaluation sets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge._text import words
from convmerge.cli import main
from convmerge.decontam import (
    DecontamStats,
    EvalIndex,
    EvalSource,
    build_index,
    decontaminate_jsonl,
)
from convmerge.recipe import RecipeError, parse_recipe, run

GSM = {
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether in April and May?",
    "answer": "Natalia sold 48/2 = 24 clips in May. #### 72",
}
KMMLU = {
    "question": "경비업법령상 시설주가 무기를 지급할 수 있는 특수경비원은?",
    "A": "민사재판에 증인으로 출석 예정인 특수경비원",
    "B": "형사사건으로 인하여 조사를 받고 있는 특수경비원",
    "C": "사의를 표명한 특수경비원",
    "D": "정신질환자인 특수경비원",
    "answer": 3,
}


def _write(path: Path, rows: list) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")  # fmt: skip
    return path


def _chat(user: str, answer: str = "ok") -> dict:
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": answer}]}  # fmt: skip


def test_words() -> None:
    assert words("Hello, World! It's 3.14") == ["hello", "world", "it", "s", "3", "14"]
    assert words("東京に行く 서울에 간다") == ["東", "京", "に", "行", "く", "서울에", "간다"]


def test_index_and_match(tmp_path: Path) -> None:
    index = EvalIndex(ngram=13, min_tokens=8)
    counts = index.add("gsm8k", ["\n".join([GSM["question"], GSM["answer"]]), "too short"])
    assert counts == {"rows": 2, "passages": 1, "too_short": 1}
    copied = "Solve: natalia SOLD clips to 48 of her friends in april, and then she sold half"
    assert index.match(copied) == (0, 1)
    labelled = "(a) Natalia sold clips to 48 of her friends in April, and then she (b) sold half"
    assert index.match(labelled) is not None
    assert index.match("Natalia sold clips to her friends in April.") is None

    short = EvalIndex(ngram=13, min_tokens=5)
    short.add("tiny", ["The quick brown fox jumps over the dog"])
    assert short.match("well, the quick brown fox jumps over the dog today") is not None
    assert short.match("the quick brown fox jumps") is None


def test_korean_choices_form_one_passage(tmp_path: Path) -> None:
    ev = _write(tmp_path / "kmmlu.jsonl", [KMMLU])
    index = build_index([EvalSource(str(ev))])
    question_only = _chat(KMMLU["question"])
    with_choices = _chat(f"{KMMLU['question']}\nA. {KMMLU['A']}\nB. {KMMLU['B']}")
    train = _write(tmp_path / "t.jsonl", [question_only, with_choices])
    st = decontaminate_jsonl(train, index)
    assert (st.kept, st.contaminated) == (1, 1)
    assert st.samples[0]["line"] == 2
    fields = build_index([EvalSource(str(ev), fields=("question",))], min_tokens=6)
    assert decontaminate_jsonl(train, fields).contaminated == 2


def test_prompts_or_all(tmp_path: Path) -> None:
    index = EvalIndex()
    index.add("gsm8k", [GSM["question"]])
    rows = [
        _chat("Tell me a story", answer=GSM["question"]),
        {"prompt": "Q?", "chosen": "fine", "rejected": GSM["question"]},
        _chat(GSM["question"]),
    ]
    train = _write(tmp_path / "t.jsonl", rows)
    assert decontaminate_jsonl(train, index).contaminated == 1
    st = DecontamStats()
    decontaminate_jsonl(train, index, check="all", output=tmp_path / "o.jsonl",
                        rejects=tmp_path / "r.jsonl", stats=st)  # fmt: skip
    assert st.contaminated == 3 and st.eval_sets["gsm8k"]["matched"] == 3
    assert (tmp_path / "o.jsonl").read_text() == ""
    assert len((tmp_path / "r.jsonl").read_text().splitlines()) == 3
    with pytest.raises(ValueError, match="check"):
        decontaminate_jsonl(train, index, check="answers")  # type: ignore[arg-type]


def test_eval_source_specs() -> None:
    assert EvalSource("hf:openai/gsm8k:main").hub == ("openai/gsm8k", "main", "test")
    assert EvalSource("hf:cais/mmlu:all:validation").hub == ("cais/mmlu", "all", "validation")
    assert EvalSource("hf:openai/openai_humaneval").hub == ("openai/openai_humaneval", None, "test")
    assert EvalSource("evals/x.jsonl").hub is None
    for bad in ("hf:", "hf:a:b:c:d"):
        with pytest.raises(ValueError, match="hf:REPO"):
            EvalSource(bad).hub  # noqa: B018


def test_hub_source(tmp_path: Path, monkeypatch) -> None:
    import convmerge.fetch.hf as hf

    calls = []

    def fake_download(repo, dst, *, config=None, split=None, token=None, max_rows=None):
        calls.append((repo, config, split))
        return _write(Path(dst), [GSM])

    monkeypatch.setattr(hf, "download_hf_dataset", fake_download)
    index = build_index([EvalSource("hf:openai/gsm8k:main")], cache_dir=tmp_path)
    assert calls == [("openai/gsm8k", "main", "test")] and index.counts[0]["passages"] == 1
    assert list(tmp_path.iterdir()) == []  # the download is removed


def test_cli(tmp_path: Path, capsys) -> None:
    ev = _write(tmp_path / "ev.jsonl", [GSM])
    train = _write(tmp_path / "t.jsonl", [_chat(GSM["question"]), _chat("hello")])
    main(["decontam", "-i", str(train), "--against", str(ev), "-o", str(tmp_path / "o.jsonl")])
    out = capsys.readouterr()
    report = json.loads(out.out)
    assert report["contaminated"] == 1 and report["eval_sets"][str(ev)]["matched"] == 1
    assert "contaminated 1 of 2" in out.err
    for argv, code in (
        (["decontam", "-i", str(tmp_path / "missing"), "--against", str(ev)], 1),
        (["decontam", "-i", str(train), "--against", str(tmp_path / "missing.jsonl")], 1),
        (["decontam", "-i", str(train), "--against", "hf:"], 2),
    ):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == code


def test_recipe_step(tmp_path: Path) -> None:
    _write(tmp_path / "ev.jsonl", [GSM])
    _write(tmp_path / "a.jsonl", [_chat(GSM["question"]), _chat("hello there")])
    raw = {
        "output": "out.jsonl",
        "sources": {"a": {"path": "a.jsonl", "convert": {"from": "auto"}}},
        "decontam": {"against": ["ev.jsonl"]},
    }
    recipe = parse_recipe(raw, path=tmp_path / "recipe.yaml")
    result = run(recipe, log=lambda _m: None)
    stats = result.report["steps"]["decontam"]["stats"]
    assert stats["contaminated"] == 1 and set(stats["eval_sets"]) == {"ev.jsonl"}
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 1
    assert run(recipe, log=lambda _m: None).ran == []
    _write(tmp_path / "ev.jsonl", [{"question": "something else entirely"}])
    assert run(recipe, log=lambda _m: None).ran == ["decontam"]
    for bad, match in (
        ({}, "decontam.against"),
        ({"against": ["hf:"]}, "hf:REPO"),
        ({"against": ["ev.jsonl"], "check": "x"}, "decontam.check"),
        ({"against": ["ev.jsonl"], "ngram": 0}, "decontam.ngram"),
        ({"against": ["ev.jsonl"], "fields": "q"}, "decontam.fields"),
    ):
        with pytest.raises(RecipeError, match=match):
            parse_recipe({**raw, "decontam": bad}, path=tmp_path / "r.yaml")
