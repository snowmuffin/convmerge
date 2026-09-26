"""Files written by convmerge 0.8.0 keep working (docs/stability.md, "Files").

``files_0_8/`` was produced by the released 0.8.0 code: the recipe project's
``recipe.lock.json``, the outputs under ``outputs/``, and the fetch completion
marker. Within a version number fields are only ever added, so every key 0.8
wrote must still be written, and the lock and marker must still be honoured.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from convmerge.cli import main

FIXTURES = Path(__file__).parent / "files_0_8"
OUTPUTS = FIXTURES / "outputs"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_keys_kept(old: dict, new: dict, where: str) -> None:
    missing = sorted(set(old) - set(new))
    assert not missing, f"{where}: keys written by 0.8 are gone: {missing}"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return Path(shutil.copytree(FIXTURES / "project", tmp_path / "project"))


def test_recipe_lock_from_0_8_is_understood(project: Path) -> None:
    pytest.importorskip("yaml")
    from convmerge import __version__
    from convmerge.recipe import load_lock, load_recipe, plan, run

    old_lock = load_lock(project / "recipe.lock.json")
    assert old_lock["steps"], "the 0.8 lock must be read, not discarded"
    recipe = load_recipe(project / "recipe.yaml")
    planned = plan(recipe)
    assert {p.step.name for p in planned} == set(old_lock["steps"])
    # Understood, not "not run before": the steps re-run because the version
    # changed (and their outputs are not part of the fixture).
    assert all(p.action == "run" and p.reason != "not run before" for p in planned)
    if __version__ != "0.8.0":
        assert planned[0].reason == f"convmerge 0.8.0 -> {__version__}"

    result = run(recipe, log=lambda _m: None)
    assert set(result.ran) == set(old_lock["steps"])
    # Same inputs, options, and seed: the same training file as 0.8.
    assert (project / "train" / "mixed.jsonl").read_bytes() == (
        OUTPUTS / "recipe_output.jsonl"
    ).read_bytes()

    new_lock = load_lock(project / "recipe.lock.json")
    _assert_keys_kept(old_lock, new_lock, "recipe.lock.json")
    for name, entry in old_lock["steps"].items():
        _assert_keys_kept(entry, new_lock["steps"][name], f"recipe.lock.json steps.{name}")
    old_report = _json(OUTPUTS / "recipe_report.json")
    new_report = _json(project / "build" / "report.json")
    _assert_keys_kept(old_report, new_report, "report.json")
    _assert_keys_kept(old_report["output"], new_report["output"], "report.json output")
    assert new_report["output"]["sha256"] == old_report["output"]["sha256"]
    for name, step in old_report["steps"].items():
        _assert_keys_kept(step, new_report["steps"][name], f"report.json steps.{name}")
        _assert_keys_kept(step["stats"], new_report["steps"][name]["stats"], name)

    assert run(recipe, log=lambda _m: None).ran == []


def test_convert_report_and_validate_output(project: Path, capsys) -> None:
    converted = project / "converted.jsonl"
    report_path = project / "report.json"
    main(
        [
            "convert", "-i", str(project / "data" / "alpaca.jsonl"), "-o", str(converted),
            "--from", "alpaca", "--format", "messages", "--report", str(report_path),
        ]
    )  # fmt: skip
    old = _json(OUTPUTS / "convert_report.json")
    new = _json(report_path)
    assert {k: new.get(k) for k in old} == old
    assert new["version"] == 1

    capsys.readouterr()
    main(["validate", "-i", str(converted)])
    old = _json(OUTPUTS / "validate.json")
    new = json.loads(capsys.readouterr().out)
    assert {k: new.get(k) for k in old} == old


def test_mix_sidecar_keys(project: Path) -> None:
    converted = project / "c.jsonl"
    main(["convert", "-i", str(project / "data" / "alpaca.jsonl"), "-o", str(converted),
          "--from", "alpaca", "--format", "messages"])  # fmt: skip
    main(["mix", "--input", f"{converted}:1", "-o", str(project / "m.jsonl"),
          "--total", "10", "--seed", "1"])  # fmt: skip
    old = _json(OUTPUTS / "mix_sidecar.mix.json")
    new = _json(project / "m.mix.json")
    _assert_keys_kept(old, new, ".mix.json")
    _assert_keys_kept(old["sources"][0], new["sources"][0], ".mix.json sources[0]")
    for key in ("version", "sampler", "seed", "total_written"):
        assert new[key] == old[key]
    for key in ("weight", "requested", "available", "written"):
        assert new["sources"][0][key] == old["sources"][0][key]


def test_fetch_marker_from_0_8_is_honoured(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("yaml")
    import convmerge.fetch.github as gh
    from convmerge.fetch import load_manifest, run_manifest

    root = Path(shutil.copytree(FIXTURES / "fetch", tmp_path / "fetch"))
    monkeypatch.chdir(root)

    def no_download(*_a: object, **_k: object) -> None:
        raise AssertionError("a completed 0.8 fetch must not be downloaded again")

    monkeypatch.setattr(gh, "download_raw_file", no_download)
    result = run_manifest(load_manifest(root / "manifest.yaml"), log=lambda _m: None)
    assert result.skipped == ["sample"] and not result.failed
