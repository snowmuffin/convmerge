"""CSV / TSV input for normalize (1.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.normalize.files import iter_data_files, normalize_path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_csv_rows_become_objects(tmp_path: Path) -> None:
    src = tmp_path / "data.csv"
    src.write_bytes(
        '﻿instruction,input,output,,output\n"Say ""hi""",,"Hi,\nthere",x,y\n\n한국어,,네\n'
        .encode()
    )  # fmt: skip
    n = normalize_path(src, tmp_path / "out.jsonl").records
    assert n == 2
    assert _rows(tmp_path / "out.jsonl") == [
        {"instruction": 'Say "hi"', "input": "", "output": "Hi,\nthere", "column_4": "x",
         "output_2": "y"},
        {"instruction": "한국어", "input": "", "output": "네", "column_4": "", "output_2": ""},
    ]  # fmt: skip


def test_tsv_and_long_fields(tmp_path: Path) -> None:
    src = tmp_path / "data.tsv"
    long = "a" * 200_000  # over the csv module's default field limit
    src.write_text(f"q\ta\nx\t{long}\n", encoding="utf-8")
    normalize_path(src, tmp_path / "out.jsonl")
    assert _rows(tmp_path / "out.jsonl") == [{"q": "x", "a": long}]


def test_bad_csv_says_where(tmp_path: Path) -> None:
    src = tmp_path / "bad.csv"
    src.write_text('a,b\n"x"y,z\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        normalize_path(src, tmp_path / "out.jsonl")


def test_directory_walks_skip_tables(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.jsonl").write_text('{"x": 1}\n')
    (tmp_path / "d" / "meta.csv").write_text("a,b\n1,2\n")
    assert [p.name for p in iter_data_files(tmp_path / "d")] == ["a.jsonl"]


def test_cli_and_convert(tmp_path: Path, capsys) -> None:
    from convmerge.cli import main

    src = tmp_path / "d.csv"
    src.write_text("instruction,output\nHi,Hello\n", encoding="utf-8")
    main(["normalize", "-i", str(src), "-o", str(tmp_path / "d.jsonl")])
    main(["convert", "--from", "auto", "-i", str(tmp_path / "d.jsonl"),
          "-o", str(tmp_path / "m.jsonl")])  # fmt: skip
    assert _rows(tmp_path / "m.jsonl")[0]["messages"][1] == {"role": "assistant",
                                                            "content": "Hello"}  # fmt: skip
