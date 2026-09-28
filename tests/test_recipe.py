"""Declarative recipes: schema, planning, incremental runs, lock file, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.recipe import (
    RECIPE_TEMPLATE,
    RecipeError,
    RecipeRunError,
    load_lock,
    load_recipe,
    parse_recipe,
    plan,
    run,
)

A = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(30)]
C = [
    {"messages": [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"r{i}"}]}
    for i in range(20)
]


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    _jsonl(tmp_path / "data" / "alpaca.jsonl", A)
    (tmp_path / "data" / "chat.json").write_text(json.dumps(C), encoding="utf-8")
    recipe = {
        "version": 1,
        "output": "train/out.jsonl",
        "sources": {
            "alpaca": {
                "path": "data/alpaca.jsonl",
                "normalize": False,
                "convert": {"from": "alpaca"},
            },
            "chat": {"path": "data/chat.json", "convert": {"from": "auto", "workers": 2}},
        },
        "mix": {"total": 40, "seed": 3, "weights": {"alpaca": 1, "chat": 1}},
        "dedupe": True,
    }
    (tmp_path / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    return tmp_path


def _quiet(_msg: str) -> None:
    pass


def test_run_then_everything_skips(project: Path) -> None:
    recipe = load_recipe(project / "recipe.json")
    first = run(recipe, log=_quiet)
    assert first.ran == ["alpaca.convert", "chat.normalize", "chat.convert", "mix", "dedupe"]
    out = (project / "train" / "out.jsonl").read_text().splitlines()
    assert len(out) == 40
    second = run(recipe, log=_quiet)
    assert second.ran == [] and len(second.skipped) == 5
    assert [p.action for p in plan(recipe)] == ["skip"] * 5
    report = json.loads((project / "build" / "report.json").read_text())
    assert report["output"]["records"] == 40
    assert report["steps"]["alpaca.convert"]["stats"]["written"] == 30
    lock = load_lock(project / "recipe.lock.json")
    assert set(lock["steps"]) == {
        "alpaca.convert",
        "chat.normalize",
        "chat.convert",
        "mix",
        "dedupe",
    }


def test_changed_input_reruns_only_its_chain(project: Path) -> None:
    recipe = load_recipe(project / "recipe.json")
    run(recipe, log=_quiet)
    _jsonl(project / "data" / "alpaca.jsonl", A[:-1])
    planned = {p.step.name: (p.action, p.reason) for p in plan(recipe)}
    assert planned["alpaca.convert"] == ("run", "input changed: data/alpaca.jsonl")
    assert planned["chat.convert"][0] == "skip"
    assert planned["mix"] == ("run", "after alpaca.convert")
    assert run(recipe, log=_quiet).ran == ["alpaca.convert", "mix", "dedupe"]


def test_rerun_with_identical_output_stops_propagation(project: Path) -> None:
    recipe = load_recipe(project / "recipe.json")
    run(recipe, log=_quiet)
    assert run(recipe, force=["normalize"], log=_quiet).ran == ["chat.normalize"]
    # Tampered output: its producer re-runs; identical result, so nothing downstream.
    with (project / "build" / "chat" / "converted.jsonl").open("a") as f:
        f.write('{"x": 1}\n')
    assert run(recipe, log=_quiet).ran == ["chat.convert"]


def test_force_patterns_and_option_changes(project: Path, tmp_path: Path) -> None:
    recipe = load_recipe(project / "recipe.json")
    run(recipe, log=_quiet)
    assert run(recipe, force=["mix"], log=_quiet).ran == ["mix"]
    assert run(recipe, force=["alpaca"], log=_quiet).ran == ["alpaca.convert"]
    assert len(run(recipe, force=[], log=_quiet).ran) == 5
    data = json.loads((project / "recipe.json").read_text())
    data["mix"]["seed"] = 4
    (project / "recipe.json").write_text(json.dumps(data))
    assert run(load_recipe(project / "recipe.json"), log=_quiet).ran == ["mix", "dedupe"]
    # Parallelism never changes output, so it never invalidates a step.
    data["sources"]["chat"]["convert"]["workers"] = 1
    (project / "recipe.json").write_text(json.dumps(data))
    assert run(load_recipe(project / "recipe.json"), log=_quiet).ran == []


def test_failed_step_keeps_previous_output_and_lock(project: Path) -> None:
    recipe = load_recipe(project / "recipe.json")
    run(recipe, log=_quiet)
    before = (project / "build" / "chat" / "converted.jsonl").read_bytes()
    lock_before = (project / "recipe.lock.json").read_text()
    (project / "data" / "chat.json").write_text("[{not json", encoding="utf-8")
    with pytest.raises(RecipeRunError, match="chat.normalize"):
        run(recipe, log=_quiet)
    assert (project / "build" / "chat" / "converted.jsonl").read_bytes() == before
    assert (
        json.loads((project / "recipe.lock.json").read_text())["steps"]
        == json.loads(lock_before)["steps"]
    )
    assert not list((project / "build" / "chat").glob(".*.part"))


def test_single_source_without_mix_copies_to_output(tmp_path: Path) -> None:
    _jsonl(tmp_path / "a.jsonl", A)
    r = parse_recipe(
        {"output": "o.jsonl", "sources": {"a": {"path": "a.jsonl", "convert": {"from": "alpaca"}}}},
        path=tmp_path / "r.yaml",
    )
    assert run(r, log=_quiet).ran == ["a.normalize", "a.convert", "output"]
    assert len((tmp_path / "o.jsonl").read_text().splitlines()) == 30


def test_fetch_source_and_manifest_reuse(tmp_path: Path, monkeypatch) -> None:
    import convmerge.fetch.github as gh

    calls: list[tuple[str, int | None]] = []

    def fake_download(url, dst, *, token=None, max_rows=None):
        calls.append((url, max_rows))
        rows = A[: max_rows or len(A)]
        return _jsonl(Path(dst), rows)

    monkeypatch.setattr(gh, "download_raw_file", fake_download)
    (tmp_path / "manifest.yaml").write_text(
        "version: 1\ndatasets:\n  - {name: remote, url: https://raw.githubusercontent.com/o/r/m/a.jsonl}\n"
    )
    r = parse_recipe(
        {
            "output": "o.jsonl",
            "sources": {
                "direct": {
                    "fetch": {
                        "url": "https://raw.githubusercontent.com/o/r/m/d.jsonl",
                        "max_rows": 5,
                    },
                    "convert": {"from": "alpaca"},
                },
                "viam": {
                    "fetch": {"manifest": "manifest.yaml", "name": "remote", "max_rows": 7},
                    "convert": {"from": "alpaca"},
                },
            },
        },
        path=tmp_path / "r.yaml",
    )
    run(r, log=_quiet)
    assert calls == [
        ("https://raw.githubusercontent.com/o/r/m/d.jsonl", 5),
        ("https://raw.githubusercontent.com/o/r/m/a.jsonl", 7),
    ]
    assert len((tmp_path / "o.jsonl").read_text().splitlines()) == 12
    assert run(r, log=_quiet).ran == []  # fetches are not repeated unless forced
    assert run(r, force=["fetch"], log=_quiet).ran == ["direct.fetch", "viam.fetch"]


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"version": 2}, "version: unsupported"),
        ({"output": None}, "output: expected a non-empty string"),
        ({"sources": {}}, "sources: at least one source"),
        ({"sources": {"bad name": {"path": "a", "convert": {"from": "auto"}}}}, "sources.bad name"),
        ({"sources": {"a": {"convert": {"from": "auto"}}}}, "exactly one of 'fetch' or 'path'"),
        ({"sources": {"a": {"path": "a"}}}, "sources.a.convert: required"),
        (
            {"sources": {"a": {"path": "a", "convert": {"from": "nope"}}}},
            "sources.a.convert.from: unknown adapter",
        ),
        (
            {"sources": {"a": {"path": "a", "convert": {"from": "auto", "format": "x"}}}},
            "unknown format",
        ),
        ({"sources": {"a": {"path": "a", "convert": {"from": "auto", "workers": 0}}}}, "workers"),
        (
            {"sources": {"a": {"path": "a", "convert": {"from": "auto", "colour": 1}}}},
            "sources.a.convert.colour: unknown key",
        ),
        (
            {
                "sources": {
                    "a": {
                        "fetch": {"url": "https://x.org/a.jsonl", "max_rows": -1},
                        "convert": {"from": "auto"},
                    }
                }
            },
            "sources.a.fetch: max_rows",
        ),
        ({"mix": {"weights": {"zzz": 1}}}, "mix.weights.zzz: no such source"),
        ({"mix": {"weights": {}}}, "mix.weights: missing weight for a"),
        ({"mix": {"sampler": "v3"}}, "mix.sampler"),
        (
            {
                "sources": {
                    "a": {
                        "path": "a",
                        "convert": {"from": "auto", "format": "preference", "preference": "chosen"},
                    }
                }
            },
            "use one or the other",
        ),
        ({"dedupe": {"keys": "messages"}}, "dedupe.keys"),
        ({"extra": 1}, "extra: unknown key"),
    ],
)
def test_schema_errors_name_the_key(tmp_path: Path, patch: dict, message: str) -> None:
    raw = {
        "version": 1,
        "output": "o.jsonl",
        "sources": {"a": {"path": "a", "convert": {"from": "auto"}}},
    }
    raw.update(patch)
    if raw.get("output") is None:
        raw["output"] = None
    with pytest.raises(RecipeError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse_recipe(raw, path=tmp_path / "r.yaml")


def test_cli_plan_frozen_init(project: Path, capsys, tmp_path: Path) -> None:
    recipe = str(project / "recipe.json")
    with pytest.raises(SystemExit) as exc:
        main(["run", recipe, "--frozen"])
    assert exc.value.code == 1
    main(["run", recipe])
    capsys.readouterr()
    main(["run", recipe, "--plan"])
    assert capsys.readouterr().out.count("skip ") == 5
    main(["run", recipe, "--frozen"])  # up to date: no exit
    main(["run", "--init", "-o", str(tmp_path / "t.yaml")])
    assert (tmp_path / "t.yaml").read_text() == RECIPE_TEMPLATE
    with pytest.raises(SystemExit) as exc:
        main(["run", str(tmp_path / "missing.yaml")])
    assert exc.value.code == 2


def test_template_is_a_valid_recipe(tmp_path: Path) -> None:
    (tmp_path / "recipe.yaml").write_text(RECIPE_TEMPLATE, encoding="utf-8")
    recipe = load_recipe(tmp_path / "recipe.yaml")
    assert list(recipe.sources) == ["alpaca", "local_chat"]
    assert recipe.dedupe is not None and recipe.mix is not None


def test_lock_prunes_removed_steps_and_files(project: Path) -> None:
    recipe = load_recipe(project / "recipe.json")
    run(recipe, log=_quiet)
    data = json.loads((project / "recipe.json").read_text())
    del data["dedupe"]
    (project / "recipe.json").write_text(json.dumps(data))
    (project / "build" / "mixed.jsonl").unlink()
    run(load_recipe(project / "recipe.json"), log=_quiet)
    lock = load_lock(project / "recipe.lock.json")
    assert "dedupe" not in lock["steps"]
    assert all((project / k).is_file() for k in lock["files"])


def test_fetch_revision_reaches_the_download(tmp_path: Path, monkeypatch) -> None:
    import convmerge.fetch.hf as hf
    import convmerge.licenses as licenses

    seen: list[str | None] = []

    def fake_download(dataset_id, dst, **kw):
        seen.append(kw.get("revision"))
        return _jsonl(Path(dst), A)

    monkeypatch.setattr(hf, "download_hf_dataset", fake_download)
    monkeypatch.setattr(licenses, "detect_hf_license", lambda *a, **k: None)
    r = parse_recipe(
        {
            "output": "o.jsonl",
            "sources": {
                "a": {
                    "fetch": {"hf": "org/ds", "revision": "abc123"},
                    "convert": {"from": "alpaca"},
                }
            },
        },
        path=tmp_path / "r.yaml",
    )
    run(r, log=_quiet)
    assert seen == ["abc123"]
