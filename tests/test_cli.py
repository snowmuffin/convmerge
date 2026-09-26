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


def test_cli_help_lists_all_install_extra(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for extra in ("[all]", "[parquet]", "[preset]", "[fetch-all]"):
        assert extra in out


def test_cli_normalize_on_json_array(tmp_path: Path) -> None:
    src = tmp_path / "in.json"
    src.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
    dst = tmp_path / "out.jsonl"
    main(["normalize", "--input", str(src), "--output", str(dst)])
    assert dst.is_file()
    assert dst.read_text(encoding="utf-8").count("\n") == 2


def test_cli_normalize_dir_skips_sidecars_and_hidden(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "repo" / ".git").mkdir(parents=True)
    (raw / "data.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (raw / "data.jsonl.fetch.json").write_text('{"version": 1}', encoding="utf-8")
    (raw / "train.mix.json").write_text('{"version": 1}', encoding="utf-8")
    (raw / "repo" / ".git" / "meta.json").write_text('{"x": 1}', encoding="utf-8")
    (raw / "repo" / "rows.json").write_text('[{"b": 2}]', encoding="utf-8")
    out = tmp_path / "out"
    main(["normalize", "--input", str(raw), "--output", str(out)])
    produced = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert produced == ["data.jsonl", "repo/rows.jsonl"]


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


def test_cli_inspect(tmp_path: Path, capsys) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(
        '{"instruction":"q","messages":[{"role":"user","content":"hi"}]}\n{"instruction":"q2"}\n',
        encoding="utf-8",
    )
    main(["inspect", "--input", str(src)])
    report = json.loads(capsys.readouterr().out)
    assert report["records"] == 2
    assert report["fields"]["instruction"]["presence"] == 1.0
    assert report["fields"]["messages"]["presence"] == 0.5
    assert set(report["fields"]["messages"]["items"].keys()) == {"role", "content"}


def test_cli_inspect_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["inspect", "--input", str(tmp_path / "nope.jsonl")])


def test_fetch_raw_shortcut_names_file_once(monkeypatch, tmp_path: Path) -> None:
    import convmerge.fetch.github as gh

    seen: dict = {}

    def fake_download(url, dst, *, token=None, max_rows=None):
        seen["dst"], seen["max_rows"] = Path(dst), max_rows
        Path(dst).write_text("{}\n", encoding="utf-8")
        return Path(dst)

    monkeypatch.setattr(gh, "download_raw_file", fake_download)
    url = "https://raw.githubusercontent.com/o/r/main/data/train.jsonl"
    main(["fetch", url, "-o", str(tmp_path), "--max-rows", "7"])
    assert seen == {"dst": tmp_path / "train.jsonl", "max_rows": 7}


def test_positive_int_options(capsys) -> None:
    for argv in (
        ["fetch", "hf://o/d", "--max-rows", "0"],
        ["convert", "-i", "a", "-o", "b", "--workers", "0"],
    ):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2
        assert "positive integer" in capsys.readouterr().err
