"""Skipped lines are reported with their reason, and normalize is suggested
only where it can help (1.6.1)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.mix import MixSource, mix_files

GOOD = json.dumps({"instruction": "q", "output": "a"})


def _write(path: Path, *lines: str) -> Path:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "bad, reason",
    [
        ('{"instruction": "x", "output":', "not valid JSON: Expecting value"),
        ("[" * 600 + "]" * 600, "nested more than 500 levels deep"),
        ('{"instruction": "\\ud800", "output": "a"}', "unpaired surrogate escape"),
    ],
)
@pytest.mark.parametrize("command", ["convert", "dedupe", "split", "filter"])
def test_warning_names_the_reason_not_normalize(bad, reason, command, tmp_path, capsys) -> None:
    src = _write(tmp_path / "in.jsonl", GOOD, bad, GOOD)
    out = str(tmp_path / "out.jsonl")
    argv = {
        "convert": ["convert", "-i", str(src), "-o", out, "--from", "auto"],
        "dedupe": ["dedupe", "-i", str(src), "-o", out],
        "split": ["split", "-i", str(src), "-o", out, "--val", "0.5"],
        "filter": ["filter", "-i", str(src), "-o", out],
    }[command]
    main(argv)
    err = capsys.readouterr().err
    assert "line 2" in err and reason in err
    assert "normalize" not in err
    assert "invalid JSON" not in err


@pytest.mark.parametrize(
    "content, shape",
    [
        (json.dumps([{"instruction": "q", "output": "a"}] * 2, indent=2), "json_array"),
        (GOOD + GOOD, "single_line"),
    ],
)
def test_normalize_is_suggested_for_shapes_it_rewrites(content, shape, tmp_path, capsys) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(content + "\n", encoding="utf-8")
    main(["dedupe", "-i", str(src), "-o", str(tmp_path / "out.jsonl")])
    err = capsys.readouterr().err
    assert f"looks like {shape}" in err and "convmerge normalize" in err


@pytest.mark.parametrize("sampler", ["v1", "v2"])
def test_mix_warns_about_skipped_lines(sampler, tmp_path, caplog) -> None:
    a = _write(tmp_path / "a.jsonl", GOOD, "{broken", GOOD, "[" * 600 + "]" * 600)
    b = _write(tmp_path / "b.jsonl", GOOD, GOOD)
    with caplog.at_level(logging.WARNING, logger="convmerge"):
        result = mix_files(
            [MixSource(a, 1), MixSource(b, 1)], tmp_path / "out.jsonl", sampler=sampler
        )
    assert result.total_written == 4
    messages = [r.getMessage() for r in caplog.records]
    assert any("a.jsonl 2 (first at line 2)" in m and "left out of the mix" in m for m in messages)
    assert not any("b.jsonl" in m for m in messages)


def test_mix_is_quiet_when_every_line_reads(tmp_path, caplog) -> None:
    a = _write(tmp_path / "a.jsonl", GOOD, GOOD)
    with caplog.at_level(logging.WARNING, logger="convmerge"):
        mix_files([MixSource(a, 1)], tmp_path / "out.jsonl")
    assert not caplog.records
