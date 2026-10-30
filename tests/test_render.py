"""convmerge render: chat-template text rows, assistant spans, CLI, and recipe step."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("jinja2")

from convmerge.cli import main  # noqa: E402
from convmerge.recipe import RecipeError, load_recipe, parse_recipe, run  # noqa: E402
from convmerge.render import RenderStats, render_jsonl  # noqa: E402
from convmerge.report import render_markdown, summarize_run  # noqa: E402
from tests.test_tokens_checks import NO_END, _tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _tokenizer(tmp_path_factory.mktemp("tok"), eos="<|im_end|>")


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _chat(*turns: tuple[str, str], **last: object) -> dict:
    msgs = [{"role": r, "content": c} for r, c in turns]
    msgs[-1].update(last)
    return {"messages": msgs}


ROWS = [
    {**_chat(("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")),
     "meta": {"source": "x"}},
    {"instruction": "hi", "input": "", "output": "yo"},
    {"messages": [{"role": "user", "content": "q"},
                  {"role": "assistant", "content": "skip", "train": False},
                  {"role": "user", "content": "q2"}, {"role": "assistant", "content": "b"}]},
    {"prompt": "p", "chosen": "c", "rejected": "r"},
    {"messages": [{"role": "user", "content": "q1"},
                  {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
                  {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]},
]  # fmt: skip


def test_text_rows_and_spans(tmp_path: Path, tok_dir: Path) -> None:
    src = _write(tmp_path / "in.jsonl", ROWS)
    plain = render_jsonl(src, tmp_path / "plain.jsonl", tokenizer=str(tok_dir))
    rows = _read(tmp_path / "plain.jsonl")
    assert (plain.rows, plain.written, plain.pairs) == (5, 4, 1)
    assert rows[0] == {
        "text": "<|im_start|> user q1 <|im_end|> <|im_start|> assistant a1 <|im_end|> "
        "<|im_start|> user q2 <|im_end|> <|im_start|> assistant a2 <|im_end|> ",
        "meta": {"source": "x"},
    }
    assert all(set(r) <= {"text", "meta"} for r in rows)

    st = RenderStats()
    render_jsonl(src, tmp_path / "spans.jsonl", tokenizer=str(tok_dir), assistant_spans=True,
                 stats=st)  # fmt: skip
    rows = _read(tmp_path / "spans.jsonl")
    trained = [[r["text"][a:b] for a, b in r["assistant_spans"]] for r in rows]
    assert trained == [["a1 <|im_end|> ", "a2 <|im_end|> "], ["yo <|im_end|> "],
                       ["b <|im_end|> "]]  # fmt: skip
    # The Qwen3-like template drops a1's reasoning once q2 follows: no prefix, no mask.
    assert (st.written, st.unstable_spans, st.pairs) == (3, 1, 1)
    assert st.to_report()["unstable_spans"] == 1


def test_template_errors_and_override(tmp_path: Path, tok_dir: Path) -> None:
    src = _write(tmp_path / "in.jsonl", [ROWS[0], "not an object"])  # type: ignore[list-item]
    st = render_jsonl(src, tmp_path / "o.jsonl", tokenizer=str(tok_dir), chat_template=NO_END)
    assert st.unreadable == 1 and _read(tmp_path / "o.jsonl")[0]["text"].startswith("user : q1")
    broken = "{{ raise_exception('no') }}"
    st = render_jsonl(src, tmp_path / "e.jsonl", tokenizer=str(tok_dir), chat_template=broken)
    assert st.written == 0 and sum(st.template_errors.values()) == 1
    with pytest.raises(ValueError, match="is the input file"):
        render_jsonl(src, src, tokenizer=str(tok_dir))


def test_cli(tmp_path: Path, tok_dir: Path, capsys) -> None:
    src = _write(tmp_path / "in.jsonl", ROWS)
    out = tmp_path / "o.jsonl"
    main(["render", "-i", str(src), "-o", str(out), "--tokenizer", str(tok_dir),
          "--assistant-spans"])  # fmt: skip
    captured = capsys.readouterr()
    assert json.loads(captured.out)["written"] == 3
    assert "--reasoning-turns last" in captured.err and "1 preference row" in captured.err
    with pytest.raises(SystemExit) as e:
        main(["render", "-i", str(tmp_path / "missing.jsonl"), "-o", str(out),
              "--tokenizer", str(tok_dir)])  # fmt: skip
    assert e.value.code == 1


def test_recipe_step(tmp_path: Path, tok_dir: Path) -> None:
    _write(tmp_path / "a.jsonl", ROWS[:3])
    raw = {
        "output": "out.jsonl",
        "sources": {"a": {"path": "a.jsonl", "convert": {"from": "auto"}}},
        "tokens": {"tokenizer": str(tok_dir)},
        "render": {"assistant_spans": True},
    }
    recipe = parse_recipe(raw, path=tmp_path / "recipe.yaml")
    assert recipe.render is not None and recipe.render.tokenizer == str(tok_dir)
    result = run(recipe, log=lambda _m: None)
    assert result.ran[-1] == "render"
    rows = _read(tmp_path / "out.jsonl")
    assert len(rows) == 3 and all("assistant_spans" in r for r in rows)
    assert run(recipe, log=lambda _m: None).ran == []
    flipped = parse_recipe({**raw, "render": True}, path=recipe.path)
    assert run(flipped, log=lambda _m: None).ran == ["render"]  # an option changed

    (tmp_path / "recipe.json").write_text(json.dumps({**raw, "render": True}))
    md = render_markdown(summarize_run(load_recipe(tmp_path / "recipe.json"), scan=False))
    assert "| render | 3 | 3 | 0 |" in md

    for bad, match in (
        ({**raw, "tokens": None, "render": True}, "render.tokenizer: required"),
        ({**raw, "render": {"assistant_spans": "yes"}}, "render.assistant_spans"),
        ({**raw, "render": {"colour": 1}}, "render.colour"),
    ):
        with pytest.raises(RecipeError, match=match):
            parse_recipe(bad, path=tmp_path / "r.yaml")
    own = parse_recipe({**raw, "tokens": None, "render": {"tokenizer": str(tok_dir)}},
                       path=tmp_path / "r.yaml")  # fmt: skip
    assert own.render is not None and own.render.local == tok_dir
