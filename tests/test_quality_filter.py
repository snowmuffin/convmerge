"""convmerge filter: rule-based quality checks, preference checks, CLI, and recipe step."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.quality import FilterSpec, FilterStats, filter_jsonl, load_rules_file
from convmerge.recipe import RecipeError, parse_recipe, run


def _chat(answer: str, user: str = "q", **extra: object) -> dict:
    msg: dict = {"role": "assistant", "content": answer, **extra}
    return {"messages": [{"role": "user", "content": user}, msg]}


def _write(path: Path, rows: list) -> Path:
    path.write_text(
        "".join((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n"
                for r in rows),
        encoding="utf-8",
    )  # fmt: skip
    return path


def _rules(tmp_path: Path, rows: list, spec: FilterSpec | None = None) -> dict[str, int]:
    st = filter_jsonl(_write(tmp_path / "in.jsonl", rows), spec=spec)
    return {k: v for k, v in st.rules.items() if v}


LOOP = " ".join(["the cat sat on the mat and then it slept well"] * 12)


def test_default_rules(tmp_path: Path) -> None:
    rows = [
        _chat("Paris is the capital of France."),
        _chat("I'm sorry, but I can't help with that request."),
        _chat("As an AI language model, I do not have opinions, but here goes."),
        _chat("죄송하지만 그 요청은 도와드릴 수 없습니다."),
        _chat(""),
        _chat(LOOP),
        _chat("ok", reasoning_content=LOOP),
        {"instruction": "q", "input": "", "output": "I cannot assist with that."},
    ]
    assert _rules(tmp_path, rows) == {
        "refusal": 4, "empty_answer": 1, "repetition": 2,
    }  # fmt: skip


def test_refusal_only_near_the_start(tmp_path: Path) -> None:
    later = (
        "Here is how. "
        + " ".join(f"Step {i} explained." for i in range(30))
        + (" I'm sorry, but I can't say more.")
    )
    discuss = "You can say 'I cannot assist with that' when declining politely."
    assert _rules(tmp_path, [_chat(later), _chat(discuss)]) == {}
    curly = _chat("I’m sorry, but I can’t do that.")
    second = _chat("Hmm. I cannot assist with that.")
    assert _rules(tmp_path, [curly, second]) == {"refusal": 2}


def test_repetition_ignores_short_and_normal_text(tmp_path: Path) -> None:
    normal = " ".join(f"word{i}" for i in range(300))
    short = "go go go go go go go go go go go go"
    assert _rules(tmp_path, [_chat(normal), _chat(short)]) == {}
    lenient = FilterSpec(repetition_max=1.0)
    assert _rules(tmp_path, [_chat(LOOP)], lenient) == {}


def test_tool_call_answers_are_not_empty(tmp_path: Path) -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    row = {"messages": [{"role": "user", "content": "q"},
                        {"role": "assistant", "content": None, "tool_calls": [call]}]}  # fmt: skip
    assert _rules(tmp_path, [row]) == {}
    assert _rules(tmp_path, [_chat("ok")], FilterSpec(min_answer_chars=5)) == {"empty_answer": 1}


def test_optional_rules(tmp_path: Path) -> None:
    sloppy = _chat("Let's delve into this. It's important to note that, in the realm of AI...")
    spec = FilterSpec.from_options(enable=["slop"], min_chars=5, max_chars=60,
                                   min_script={"hangul": 0.5})  # fmt: skip
    rows = [
        sloppy,
        _chat("짧음"),
        _chat("This answer is in English only, not Korean at all."),
        _chat("한국어로 된 충분히 긴 답변입니다 `code` 포함", user="한국어 질문입니다"),
        _chat("코드만 영어입니다.\n```python\nprint('hello world, english code')\n```",
              user="파이썬 코드를 보여 주세요"),
    ]  # fmt: skip
    assert _rules(tmp_path, rows, spec) == {"slop": 1, "length": 2, "script": 2}
    assert set(spec.active) == {*spec.rules}
    with pytest.raises(ValueError, match="length rule needs"):
        FilterSpec(rules=("length",))
    with pytest.raises(ValueError, match="unknown script"):
        FilterSpec(min_script={"klingon": 0.5})
    with pytest.raises(ValueError, match="unknown filter rule"):
        FilterSpec(rules=("nope",))


def test_preference_checks(tmp_path: Path) -> None:
    rows = [
        {"prompt": "p", "chosen": "Answer A", "rejected": "answer   a"},
        {"prompt": "p", "chosen": "I'm sorry, but I can't.", "rejected": "Sure: 42"},
        {"prompt": "p", "chosen": "Fine answer", "rejected": "I'm sorry, but I can't."},
        {"prompt": "p", "chosen": "Fine answer", "rejected": ""},
    ]
    st = filter_jsonl(_write(tmp_path / "p.jsonl", rows))
    assert {k: v for k, v in st.rules.items() if v} == {
        "near_identical_pair": 1, "refusal": 1, "rejected_empty": 1,
    }  # fmt: skip
    assert st.kept == 1 and st.pairs == 4


def test_length_bias_warning(tmp_path: Path) -> None:
    rows = [{"prompt": f"p{i}", "chosen": "a much longer chosen answer", "rejected": "short"}
            for i in range(20)]  # fmt: skip
    st = filter_jsonl(_write(tmp_path / "p.jsonl", rows))
    pref = st.to_report()["preference"]
    assert pref["chosen_longer_share"] == 1.0 and pref["median_length_ratio"] == 5.4
    assert "length rather than quality" in st.warnings()[0]


def test_patterns_and_rules_file(tmp_path: Path) -> None:
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({
        "refusal": ["i would rather not"],
        "slop": ["synergy"],
        "patterns": {"url": "https?://"},
    }))  # fmt: skip
    spec = FilterSpec(**load_rules_file(rules))
    rows = [_chat("I would rather not answer."), _chat("See https://example.com")]
    assert _rules(tmp_path, rows, spec) == {"refusal": 1, "url": 1}
    assert _rules(tmp_path, [_chat("As an AI language model, sure.")],
                  FilterSpec(builtin_phrases=False)) == {}  # fmt: skip
    rules.write_text(json.dumps({"patterns": {"refusal": "x"}}))
    with pytest.raises(ValueError, match="built-in rule"):
        FilterSpec(**load_rules_file(rules))
    rules.write_text(json.dumps({"bogus": 1}))
    with pytest.raises(ValueError, match="unknown key"):
        load_rules_file(rules)


def test_output_is_byte_for_byte(tmp_path: Path) -> None:
    raw = '{"messages": [{"role": "user", "content": "q"},  {"role": "assistant", "content": "a"}]}'
    src = _write(tmp_path / "in.jsonl", [raw, _chat(""), "{not json", {"foo": 1}])
    st = FilterStats()
    filter_jsonl(src, output=tmp_path / "o.jsonl", rejects=tmp_path / "r.jsonl", stats=st)
    assert (tmp_path / "o.jsonl").read_text() == raw + "\n"
    assert len((tmp_path / "r.jsonl").read_text().splitlines()) == 2
    assert (st.rows, st.kept, st.rejected, st.unreadable, st.invalid_json) == (3, 1, 2, 1, 1)


def test_cli(tmp_path: Path, capsys) -> None:
    src = _write(tmp_path / "in.jsonl", [_chat("fine"), _chat("I'm sorry, but I can't.")])
    main(["filter", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--enable", "slop",
          "--min-script", "latin=0.5"])  # fmt: skip
    out = capsys.readouterr()
    report = json.loads(out.out)
    assert report["version"] == 1 and report["kept"] == 1
    assert set(report["rules"]) >= {"refusal", "slop", "script"}
    assert "rejected 1 of 2" in out.err
    main(["filter", "-i", str(src), "--disable", "refusal"])
    assert json.loads(capsys.readouterr().out)["rejected"] == 0
    for argv, code in (
        (["filter", "-i", str(tmp_path / "missing.jsonl")], 1),
        (["filter", "-i", str(src), "--min-script", "hangul"], 2),
        (["filter", "-i", str(src), "--rules-file", str(tmp_path / "missing.yaml")], 2),
    ):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == code


def test_recipe_step(tmp_path: Path) -> None:
    _write(tmp_path / "a.jsonl", [_chat("fine"), _chat("I'm sorry, but I can't."), _chat(LOOP)])
    (tmp_path / "rules.json").write_text(json.dumps({"patterns": {"fine": "^fine$"}}))
    raw = {
        "output": "out.jsonl",
        "sources": {"a": {"path": "a.jsonl", "convert": {"from": "auto"}}},
        "filter": {"disable": ["repetition"], "rules_file": "rules.json"},
    }
    recipe = parse_recipe(raw, path=tmp_path / "recipe.yaml")
    result = run(recipe, log=lambda _m: None)
    stats = result.report["steps"]["filter"]["stats"]
    assert stats["rules"] == {"empty_answer": 0, "refusal": 1, "near_identical_pair": 0,
                              "rejected_empty": 0, "fine": 1}  # fmt: skip
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 1
    assert run(recipe, log=lambda _m: None).ran == []
    (tmp_path / "rules.json").write_text(json.dumps({"patterns": {}}))
    assert run(recipe, log=lambda _m: None).ran == ["filter"]

    assert parse_recipe({**raw, "filter": True}, path=tmp_path / "r.yaml").filter is not None
    for bad, match in (
        ({"enable": ["nope"]}, "unknown filter rule"),
        ({"repetition_max": "x"}, "filter.repetition_max"),
        ({"min_script": {"hangul": 2}}, "min_script.hangul"),
        ({"colour": 1}, "filter.colour: unknown key"),
    ):
        with pytest.raises(RecipeError, match=match):
            parse_recipe({**raw, "filter": bad}, path=tmp_path / "r.yaml")
