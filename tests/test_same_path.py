"""An output that names the input file is refused before anything is written (1.6.1)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.io import SamePathError, refuse_overwrite

ROWS = 5


def _write(path: Path) -> Path:
    rows = [
        {"messages": [{"role": "user", "content": f"q{i}"},
                      {"role": "assistant", "content": f"a{i}"}]}
        for i in range(ROWS)
    ]  # fmt: skip
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize(
    "argv",
    [
        ["convert", "-i", "{x}", "-o", "{x}", "--from", "auto"],
        ["dedupe", "-i", "{x}", "-o", "{x}"],
        ["dedupe", "-i", "{x}", "-o", "{y}", "--rejects", "{x}"],
        ["dedupe", "-i", "{x}", "-o", "{y}", "--rejects", "{y}"],
        ["dedupe", "--near", "-i", "{x}", "-o", "{x}"],
        ["filter", "-i", "{x}", "-o", "{x}"],
        ["decontam", "-i", "{x}", "-o", "{x}", "--against", "{x}"],
        ["split", "-i", "{x}", "-o", "{x}", "--val", "0.5"],
        ["split", "-i", "{x}", "-o", "{y}", "--val", "0.5", "--val-output", "{x}"],
        ["turns", "-i", "{x}", "--single-out", "{x}", "--multi-out", "{y}"],
        ["normalize", "-i", "{x}", "-o", "{x}"],
        ["normalize", "-i", "{dir}", "-o", "{dir}"],
    ],
)
def test_cli_refuses_to_overwrite_its_input(argv, tmp_path: Path, capsys) -> None:
    x = _write(tmp_path / "x.jsonl")
    y = tmp_path / "y.jsonl"
    args = [a.format(x=x, y=y, dir=tmp_path) for a in argv]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "error: output" in err and "Traceback" not in err
    assert _lines(x) == ROWS


def test_symlinked_output_counts_as_the_input(tmp_path: Path) -> None:
    x = _write(tmp_path / "x.jsonl")
    link = tmp_path / "link.jsonl"
    try:
        os.symlink(x, link)
    except OSError:
        pytest.skip("symlinks not available")
    with pytest.raises(SamePathError, match="is the input file"):
        refuse_overwrite([x], [link])


def test_api_functions_refuse_too(tmp_path: Path) -> None:
    from convmerge.convert import convert_file
    from convmerge.normalize.dedup import deduplicate_jsonl
    from convmerge.split import split_jsonl
    from convmerge.tokens import check_tokens

    x = _write(tmp_path / "x.jsonl")
    calls = [
        lambda: convert_file(x, x, adapter_name="auto", output_format="messages"),
        lambda: deduplicate_jsonl(x, x),
        lambda: split_jsonl(x, tmp_path / "t.jsonl", x, val=0.5),
        # Checked before the tokenizer is loaded, so this needs no transformers.
        lambda: check_tokens(x, tokenizer="unused", output=x),
    ]
    for call in calls:
        with pytest.raises(SamePathError):
            call()
    assert _lines(x) == ROWS


def test_distinct_paths_pass(tmp_path: Path) -> None:
    x = _write(tmp_path / "x.jsonl")
    refuse_overwrite([x], [tmp_path / "y.jsonl", None, tmp_path / "sub" / "x.jsonl"])
    main(["dedupe", "-i", str(x), "-o", str(tmp_path / "y.jsonl")])
    assert _lines(tmp_path / "y.jsonl") == ROWS
