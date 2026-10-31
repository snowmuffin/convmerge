"""The recipe label: step (one label.NAME step per entry) and its report section."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from convmerge.recipe import RecipeError, RecipeRunError, load_recipe, plan, run
from convmerge.report import dataset_card, render_html, render_markdown, summarize_run
from tests.test_llm import FakeServer


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[FakeServer, str]]:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    state = FakeServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), state.handler())
    threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True).start()
    yield state, f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def _reply(body: dict) -> str:
    text = body["messages"][-1]["content"].split("Conversation:", 1)[1]
    if "on a scale" in body["messages"][-1]["content"]:
        return str(min(5, 1 + text.count("x")))
    return "math" if "sum" in text else "other"


def _project(tmp_path: Path, url: str, **extra: object) -> Path:
    rows = [{"messages": [{"role": "user", "content": f"sum {'x' * i}" if i % 2 else f"hi {i}"},
                          {"role": "assistant", "content": "ok"}]} for i in range(6)]  # fmt: skip
    (tmp_path / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "rubric.md").write_text("More x is harder.")
    recipe = {
        "version": 1,
        "output": "out.jsonl",
        "sources": {"a": {"path": "a.jsonl", "convert": {"from": "auto"}}},
        "label": [
            {"name": "topic", "endpoint": url, "model": "tiny", "classes": ["math", "other"],
             "question": "Topic?"},
            {"name": "difficulty", "endpoint": url, "model": "tiny", "scale": "1-5",
             "rubric": "rubric.md", "workers": 2},
        ],
        "order": {"by": "meta:difficulty"},
        **extra,
    }  # fmt: skip
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    return path


SOURCE = {"path": "a.jsonl", "convert": {"from": "auto"}}


def _quiet(_msg: str) -> None:
    pass


def test_label_steps_run_in_place_and_rerun_only_when_needed(
    fake: tuple[FakeServer, str], tmp_path: Path
) -> None:
    server, url = fake
    server.reply = _reply
    path = _project(tmp_path, url)
    first = run(load_recipe(path), log=_quiet)
    assert first.ran == ["a.normalize", "a.convert", "label.topic", "label.difficulty", "order"]
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [r["meta"]["difficulty"] for r in rows] == sorted(r["meta"]["difficulty"] for r in rows)
    assert {r["meta"]["topic"] for r in rows} == {"math", "other"}
    assert len(server.requests) == 12
    lock = json.loads((tmp_path / "recipe.lock.json").read_text())
    options = lock["steps"]["label.difficulty"]["options"]
    assert options["host"] == url.split("/")[2] and options["rubric"] == "rubric.md"
    assert not {"workers", "endpoint", "api_key_env", "max_calls"} & set(options)
    assert (tmp_path / "build" / ".convmerge-cache" / "llm.sqlite").is_file()

    assert run(load_recipe(path), log=_quiet).ran == []
    # workers, timeout and the budget do not change the output: nothing re-runs.
    recipe = json.loads(path.read_text())
    recipe["label"][0].update(workers=1, timeout=5, max_calls=100)
    path.write_text(json.dumps(recipe))
    assert run(load_recipe(path), log=_quiet).ran == []
    # A changed rubric re-runs that label step; its downstream order re-runs only if
    # the labels changed. The new prompt is not cached, so it calls again.
    (tmp_path / "rubric.md").write_text("More x is harder. 5 is the hardest.")
    actions = {p.step.name: p.reason for p in plan(load_recipe(path)) if p.action == "run"}
    assert actions == {"label.difficulty": "input changed: rubric.md",
                       "order": "after label.difficulty"}  # fmt: skip
    again = run(load_recipe(path), log=_quiet)
    assert again.ran == ["label.difficulty"] and len(server.requests) == 18
    # Forcing a re-run answers from the cache: no new calls.
    forced = run(load_recipe(path), force=["label.topic"], log=_quiet)
    assert forced.ran == ["label.topic"] and len(server.requests) == 18
    report = json.loads((tmp_path / "build" / "report.json").read_text())
    stats = report["steps"]["label.topic"]["stats"]
    assert (stats["cache_hits"], stats["calls"], stats["labeled"]) == (6, 0, 6)


def test_label_step_failure_names_the_step(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    server.status = [404]
    path = _project(tmp_path, url)
    with pytest.raises(RecipeRunError, match=r"label\.topic: LLMError: .*HTTP 404"):
        run(load_recipe(path), log=_quiet)
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"endpoint": "https://api.example.com/v1"}, "not a local or private host"),
        ({"scale": "1-5"}, "exactly one of 'classes' or 'scale'"),
        ({"name": "a.b"}, "letters, digits"),
        ({"scale": "5", "classes": None}, "scale: expected LO-HI"),
        ({"workers": 0}, "workers: expected an integer of at least 1"),
        ({"temperature": 1}, "temperature: unknown key"),
        ({"endpoint": "localhost:8000"}, "expected an http"),
    ],
)
def test_label_schema_errors(tmp_path: Path, entry: dict, message: str) -> None:
    label = {"name": "topic", "endpoint": "http://localhost:8000/v1", "model": "m",
             "classes": ["a", "b"], "question": "q", **entry}  # fmt: skip
    label = {k: v for k, v in label.items() if v is not None}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"version": 1, "output": "o.jsonl",
                                "sources": {"a": SOURCE}, "label": [label]}))  # fmt: skip
    with pytest.raises(RecipeError, match=message):
        load_recipe(path)


def test_label_schema_names_and_external_hosts(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"

    def load(labels: object, sources: dict | None = None) -> object:
        path.write_text(json.dumps({"version": 1, "output": "o.jsonl",
                                    "sources": sources or {"a": SOURCE},
                                    "label": labels}))  # fmt: skip
        return load_recipe(path)

    entry = {"name": "convert", "endpoint": "http://10.0.0.2:8000/v1", "model": "m",
             "classes": "a,b", "question": "q"}  # fmt: skip
    assert load([entry]).label[0].classes == ("a", "b")  # type: ignore[attr-defined]
    with pytest.raises(RecipeError, match="used twice"):
        load([entry, entry])
    with pytest.raises(RecipeError, match="share its name with the source 'label'"):
        load([entry], {"label": SOURCE})
    with pytest.raises(RecipeError, match="non-empty list"):
        load({"name": "x"})
    external = {**entry, "endpoint": "https://api.example.com/v1", "allow_external": True,
                "api_key_env": "SOME_KEY"}  # fmt: skip
    spec = load([external]).label[0]  # type: ignore[attr-defined]
    assert spec.allow_external and spec.api_key_env == "SOME_KEY"


def test_report_shows_label_distributions(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    server.reply = _reply
    path = _project(tmp_path, url)
    run(load_recipe(path), log=_quiet)
    summary = summarize_run(load_recipe(path))
    steps = {s["name"]: s for s in summary["steps"]}
    assert (steps["label.topic"]["rows_in"], steps["label.topic"]["removed"]) == (6, 0)
    assert steps["label.topic"]["note"] == "6 labeled into meta.topic"
    labels = {lab["field"]: lab for lab in summary["labels"]}
    assert list(labels) == ["meta.topic", "meta.difficulty"]
    topic = labels["meta.topic"]
    assert [(v["value"], v["rows"]) for v in topic["values"]] == [("math", 3), ("other", 3)]
    assert topic["served_models"] == ["tiny-served"] and topic["calls"] == 6
    assert [v["value"] for v in labels["meta.difficulty"]["values"]] == ["1", "2", "3", "4", "5"]
    md = render_markdown(summary)
    assert "## Labels: `meta.topic`" in md and "| math | 3 | 50.0% |" in md
    assert "classes by model tiny (served as tiny-served) at 127.0.0.1" in md
    page = render_html(summary)
    assert "<h2>Labels: <code>meta.difficulty</code></h2>" in page
    card = dataset_card(summary)
    assert "## Model labels" in card and "`meta.topic`:" in card and "math 50%" in card


def test_report_counts_rows_without_a_label(fake: tuple[FakeServer, str], tmp_path: Path) -> None:
    server, url = fake
    server.reply = lambda body: "neither"  # never parses
    path = _project(tmp_path, url)
    recipe = json.loads(path.read_text())
    del recipe["label"][1], recipe["order"]
    path.write_text(json.dumps(recipe))
    run(load_recipe(path), log=_quiet)
    lab = summarize_run(load_recipe(path))["labels"][0]
    assert lab["values"][-1] == {"value": "(no label)", "rows": 6, "share": 1.0}
    assert lab["parse_failures"] == 6
    assert "6 parse failures" in render_markdown(summarize_run(load_recipe(path)))
