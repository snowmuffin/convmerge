"""convmerge report: readable build reports and dataset cards from a recipe run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.recipe import load_recipe, run
from convmerge.report import dataset_card, render_html, render_markdown, summarize_run


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _project(tmp_path: Path, **extra: object) -> Path:
    _write(tmp_path / "a.jsonl", [
        {"instruction": f"task {i % 5}", "input": "", "output": ans}
        for i, ans in enumerate(["Paris.", "", "I'm sorry, but I can't help with that.",
                                 "A <b>bold</b> | piped answer. " * 3, "Paris."] * 4)
    ])  # fmt: skip
    _write(tmp_path / "b.jsonl", [
        {"conversations": [{"from": "human", "value": f"q{i}-{t}"} if k == 0 else
                           {"from": "gpt", "value": " ".join(f"w{j}" for j in range(i * 40 + 1))}
                           for t in range(i % 3 + 1) for k in range(2)]}
        for i in range(6)
    ])  # fmt: skip
    _write(tmp_path / "eval.jsonl", [{"prompt": "task 0"}])
    recipe = {
        "version": 1,
        "output": "out.jsonl",
        "sources": {
            "alpaca_like": {"path": "a.jsonl", "license": "cc-by-nc-4.0",
                            "convert": {"from": "auto"}},
            "chat": {"path": "b.jsonl", "license": "apache-2.0", "convert": {"from": "auto"}},
        },
        "mix": {"weights": {"alpaca_like": 1, "chat": 1}, "seed": 0},
        "dedupe": True,
        "filter": True,
        "decontam": {"against": ["eval.jsonl"], "min_tokens": 1},
        **extra,
    }  # fmt: skip
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    run(load_recipe(path), log=lambda _m: None)
    return path


def test_summary_steps_sources_and_shape(tmp_path: Path) -> None:
    summary = summarize_run(load_recipe(_project(tmp_path)))
    by_name = {s["name"]: s for s in summary["sources"]}
    assert by_name["alpaca_like"]["origin"] == "a.jsonl"
    assert by_name["alpaca_like"]["warning"] == "non-commercial or research-only terms"
    assert (by_name["alpaca_like"]["read"], by_name["alpaca_like"]["converted"]) == (20, 16)
    steps = {s["name"]: s for s in summary["steps"]}
    assert list(steps) == ["alpaca_like.convert", "chat.convert", "mix", "dedupe", "filter",
                           "decontam"]  # fmt: skip
    assert steps["alpaca_like.convert"]["reasons"] == {"empty_answer": 4}
    assert steps["dedupe"]["reasons"] == {"duplicates": 12}  # four copies of three rows
    assert steps["filter"]["reasons"]["refusal"] == 1
    assert set(steps["decontam"]["reasons"]) == {"contaminated"}  # rows asking "task 0"
    for st in steps.values():  # every step accounts for its rows
        assert st["rows_in"] - st["rows_out"] == st["removed"]
    shape = summary["shape"]
    out_rows = len((tmp_path / "out.jsonl").read_text().splitlines())
    assert shape["rows"] == out_rows == summary["output"]["records"]
    assert shape["layout"] == "messages"
    assert sum(shape["turns"].values()) == sum(shape["answer_histogram"].values()) == out_rows
    assert shape["turns"]["3"] >= 1 and shape["answer_chars"]["max"] >= 200
    assert summarize_run(load_recipe(tmp_path / "recipe.json"), scan=False)["shape"] is None


def test_renderers_escape_and_card_front_matter(tmp_path: Path) -> None:
    summary = summarize_run(load_recipe(_project(tmp_path)))
    md = render_markdown(summary)
    assert "| alpaca_like | a.jsonl | cc-by-nc-4.0 ⚠ |" in md
    assert "- alpaca_like (cc-by-nc-4.0): non-commercial or research-only terms" in md
    assert "| dedupe | 22 | 10 | 12 | duplicates 12 |" in md
    assert "| filter | 10 | 9 | 1 | refusal 1 |" in md  # one rule fired: no overlap note
    page = render_html(summary)
    assert page.startswith("<!doctype html>") and "<script" not in page
    assert "<b>bold</b>" not in page  # row text is escaped
    card = dataset_card(summary, title="My mix")
    front = card.split("---")[1]
    assert "license: other" in front and "- n<1K" in front and "- convmerge" in front
    assert "# My mix" in card and "convmerge run recipe.json" in card
    assert summary["output"]["sha256"] in card

    one = {**summary, "sources": [s for s in summary["sources"] if s["name"] == "chat"],
           "license_warnings": []}  # fmt: skip
    assert "license: apache-2.0" in dataset_card(one)
    unknown = {**one, "sources": [{**one["sources"][0], "license": None}]}
    assert "license: other" in dataset_card(unknown)


def test_split_steps(tmp_path: Path) -> None:
    path = _project(tmp_path, split={"val_rows": 2, "val_output": "val.jsonl"})
    steps = {s["name"]: s for s in summarize_run(load_recipe(path))["steps"]}
    assert steps["split.val"]["rows_out"] == 2 and steps["split.val"]["removed"] is None
    assert steps["split.train"]["rows_out"] == steps["split.train"]["rows_in"] - 2


def test_cli(tmp_path: Path, capsys) -> None:
    recipe = _project(tmp_path)
    main(["report", str(recipe)])
    assert capsys.readouterr().out.startswith("# Build report: out.jsonl")
    main(["report", str(recipe), "--html", str(tmp_path / "r.html"),
          "--markdown", str(tmp_path / "r.md"), "--card", str(tmp_path / "README.md"),
          "--title", "demo", "--no-scan"])  # fmt: skip
    assert "## Output" not in (tmp_path / "r.md").read_text()
    assert "# demo" in (tmp_path / "README.md").read_text()
    assert (tmp_path / "r.html").read_text().startswith("<!doctype html>")

    with pytest.raises(SystemExit) as e:  # the report must not replace a recipe input
        main(["report", str(recipe), "--card", str(tmp_path / "out.jsonl")])
    assert e.value.code == 2
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "recipe.json").write_text(recipe.read_text())
    with pytest.raises(SystemExit) as e:
        main(["report", str(fresh / "recipe.json")])
    assert e.value.code == 1 and "run the recipe first" in capsys.readouterr().err
