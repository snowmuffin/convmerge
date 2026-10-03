"""Invalid values are refused with a message, and results that are probably not
what was meant are explained (1.6.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import main

ROW = json.dumps({"messages": [{"role": "user", "content": "q"},
                               {"role": "assistant", "content": "a"}]})  # fmt: skip


def _rows(path: Path, n: int = 5) -> Path:
    path.write_text((ROW + "\n") * n, encoding="utf-8")
    return path


def _exit(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return int(exc.value.code or 0)


def test_validate_explains_skipped_lines(tmp_path, capsys) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(ROW + "\n{bad\n", encoding="utf-8")
    assert _exit(["validate", "-i", str(src)]) == 1
    err = capsys.readouterr().err
    assert "skipped 1 line" in err and "line 2 (not valid JSON" in err


def test_validate_suggests_normalize_for_a_json_array(tmp_path, capsys) -> None:
    src = tmp_path / "in.json"
    src.write_text(json.dumps([json.loads(ROW)] * 2), encoding="utf-8")
    assert _exit(["validate", "-i", str(src)]) == 1
    assert "convmerge normalize" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect", "-i", "{x}", "--max-rows", "0"],
        ["inspect", "-i", "{x}", "--max-examples", "-1"],
        ["mix", "-i", "{x}:1", "-o", "{o}", "-n", "0"],
        ["mix", "-i", "{x}:0", "-o", "{o}"],
        ["mix", "-i", "{x}:-1", "-i", "{x}:2", "-o", "{o}", "-n", "4"],
        ["filter", "-i", "{x}", "--min-chars", "10", "--max-chars", "5"],
        ["filter", "-i", "{x}", "--rules-file", "{missing}"],
        ["convert", "-i", "{x}", "-o", "{o}", "--preset", "{missing}"],
        ["run", "{missing}"],
        ["fetch", "ftp://example.com/data", "-o", "{dir}"],
    ],
)
def test_invalid_values_exit_2_with_a_message(argv, tmp_path, capsys) -> None:
    x = _rows(tmp_path / "x.jsonl")
    args = [a.format(x=x, o=tmp_path / "o.jsonl", missing=tmp_path / "nope.yaml", dir=tmp_path)
            for a in argv]  # fmt: skip
    assert _exit(args) == 2
    err = capsys.readouterr().err
    assert "error" in err and "Traceback" not in err and "Errno" not in err


def test_mix_config_total_must_be_positive(tmp_path, capsys) -> None:
    x = _rows(tmp_path / "x.jsonl")
    cfg = tmp_path / "mix.json"
    cfg.write_text(json.dumps({"sources": [{"path": str(x), "weight": 1}], "total": 0,
                               "output": str(tmp_path / "o.jsonl")}), encoding="utf-8")  # fmt: skip
    assert _exit(["mix", str(cfg), "--no-recipe"]) == 2
    assert "'total' in the config" in capsys.readouterr().err


def test_filter_spec_rejects_min_above_max() -> None:
    from convmerge.quality import FilterSpec

    with pytest.raises(ValueError, match="min_chars 10 is greater than max_chars 5"):
        FilterSpec.from_options(min_chars=10, max_chars=5)


def test_mix_files_rejects_negative_weights(tmp_path) -> None:
    from convmerge.mix import MixSource, mix_files

    x = _rows(tmp_path / "x.jsonl")
    with pytest.raises(ValueError, match="must not be negative"):
        mix_files([MixSource(x, -1), MixSource(x, 2)], tmp_path / "o.jsonl")


def test_fetch_only_names_must_exist(tmp_path, capsys) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text("datasets:\n  - name: alpha\n    hf: org/ds\n", encoding="utf-8")
    pytest.importorskip("yaml")
    assert _exit(["fetch", str(manifest), "--only", "alpah", "-o", str(tmp_path)]) == 2
    assert "'alpah' (did you mean 'alpha'?)" in capsys.readouterr().err


def test_fetch_warns_about_an_empty_manifest(tmp_path, capsys) -> None:
    pytest.importorskip("yaml")
    manifest = tmp_path / "m.yaml"
    manifest.write_text("datasets: []\n", encoding="utf-8")
    main(["fetch", str(manifest), "-o", str(tmp_path / "raw")])
    assert "lists no datasets" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--val-rows", "50"], "the train file is empty: --val-rows 50"),
        (["--val", "0.99"], "the train file is empty: --val 0.99"),
        (["--val", "0.01"], "the validation file is empty: --val 0.01 of 5 rows"),
    ],
)
def test_split_warns_about_an_empty_side(extra, message, tmp_path, capsys) -> None:
    x = _rows(tmp_path / "x.jsonl")
    main(["split", "-i", str(x), "-o", str(tmp_path / "t.jsonl"), *extra])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["run", "--init", "-o", "{f}"], ["preset", "init", "-o", "{f}"]])
def test_init_templates_do_not_replace_existing_files(argv, tmp_path, capsys) -> None:
    mine = tmp_path / "mine.yaml"
    mine.write_text("hand written\n", encoding="utf-8")
    assert _exit([a.format(f=mine) for a in argv]) == 2
    assert "already exists" in capsys.readouterr().err
    assert mine.read_text(encoding="utf-8") == "hand written\n"
    fresh = tmp_path / "fresh.yaml"
    main([a.format(f=fresh) for a in argv])
    assert fresh.is_file()
