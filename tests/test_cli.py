"""Smoke tests for CLI subcommands (no network, no heavy deps)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import main


def test_cli_help_runs(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "convmerge" in out
    for cmd in ("convert", "normalize", "dedupe", "turns", "fetch"):
        assert cmd in out


def test_cli_normalize_on_json_array(tmp_path: Path) -> None:
    src = tmp_path / "in.json"
    src.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
    dst = tmp_path / "out.jsonl"
    main(["normalize", "--input", str(src), "--output", str(dst)])
    assert dst.is_file()
    assert dst.read_text(encoding="utf-8").count("\n") == 2


def test_cli_dedupe(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text('{"x":1}\n{"x":1}\n{"x":2}\n', encoding="utf-8")
    dst = tmp_path / "out.jsonl"
    main(["dedupe", "--input", str(src), "--output", str(dst)])
    assert dst.read_text(encoding="utf-8").count("\n") == 2


def test_cli_turns(tmp_path: Path, capsys) -> None:
    src = tmp_path / "in.jsonl"
    sample_single = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    }
    sample_multi = {
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
    }
    src.write_text(
        json.dumps(sample_single) + "\n" + json.dumps(sample_multi) + "\n",
        encoding="utf-8",
    )
    single_out = tmp_path / "single.jsonl"
    multi_out = tmp_path / "multi.jsonl"
    main(
        [
            "turns",
            "--input",
            str(src),
            "--single-out",
            str(single_out),
            "--multi-out",
            str(multi_out),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["single"] == 1
    assert report["multi"] == 1
    assert single_out.is_file() and multi_out.is_file()


def test_cli_fetch_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["fetch", str(tmp_path / "does_not_exist.yaml")])
