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


def _workbook(path: Path, *, formula_values: bool = False) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    import datetime

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "data"
    ws.append(["instruction", "input", "output", None, "output", None, None])
    ws.append(["Add", 1, 3.0, True, "=1+2"])
    ws.append([datetime.datetime(2024, 1, 2, 3, 4), datetime.date(2024, 5, 6), 0.1 + 0.2])
    ws.append([None, None, None])
    ws.append(["multi\nline", None, "한국어", None, None, None, "extra"])
    wb.create_sheet("second").append(["q", "a"])
    wb["second"].append(["hello", "world"])
    wb.save(path)
    if formula_values:  # what Excel stores on save: the formula and its value
        import zipfile

        with zipfile.ZipFile(path) as z:
            parts = {n: z.read(n) for n in z.namelist()}
        sheet = "xl/worksheets/sheet1.xml"
        assert b"<f>1+2</f><v />" in parts[sheet]
        parts[sheet] = parts[sheet].replace(b"<f>1+2</f><v />", b"<f>1+2</f><v>3</v>")
        with zipfile.ZipFile(path, "w") as z:
            for n, data in parts.items():
                z.writestr(n, data)
    return path


def test_xlsx_rows_become_objects(tmp_path: Path, caplog) -> None:
    src = _workbook(tmp_path / "data.xlsx")
    with caplog.at_level("WARNING"):
        n = normalize_path(src, tmp_path / "out.jsonl").records
    assert n == 3
    assert _rows(tmp_path / "out.jsonl") == [
        {"instruction": "Add", "input": "1", "output": "3", "column_4": "TRUE", "output_2": ""},
        {"instruction": "2024-01-02 03:04:00", "input": "2024-05-06", "output": "0.3",
         "column_4": "", "output_2": ""},
        {"instruction": "multi\nline", "input": "", "output": "한국어", "column_4": "",
         "output_2": "", "column_6": "", "column_7": "extra"},
    ]  # fmt: skip
    assert "1 formula cell(s) have no saved value" in caplog.text
    normalize_path(src, tmp_path / "second.jsonl", sheet="second")
    assert _rows(tmp_path / "second.jsonl") == [{"q": "hello", "a": "world"}]


def test_xlsx_formula_with_saved_value(tmp_path: Path, caplog) -> None:
    src = _workbook(tmp_path / "data.xlsx", formula_values=True)
    with caplog.at_level("WARNING"):
        normalize_path(src, tmp_path / "out.jsonl")
    assert _rows(tmp_path / "out.jsonl")[0]["output_2"] == "3"
    assert "formula" not in caplog.text


def test_xlsx_errors(tmp_path: Path, capsys, monkeypatch) -> None:
    from convmerge.cli import main

    src = _workbook(tmp_path / "data.xlsx")
    with pytest.raises(ValueError, match="no sheet 'nope'; sheets: data, second"):
        normalize_path(src, tmp_path / "o.jsonl", sheet="nope")
    (tmp_path / "old.xls").write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(SystemExit) as e:
        main(["normalize", "-i", str(tmp_path / "old.xls"), "-o", str(tmp_path / "o.jsonl")])
    assert e.value.code == 1 and "save it as .xlsx or .csv" in capsys.readouterr().err
    (tmp_path / "bad.xlsx").write_bytes(b"not a zip")
    with pytest.raises(SystemExit) as e:
        main(["normalize", "-i", str(tmp_path / "bad.xlsx"), "-o", str(tmp_path / "o.jsonl")])
    assert e.value.code == 1 and "not a readable .xlsx workbook" in capsys.readouterr().err
    with pytest.raises(ValueError, match="sheet applies to .xlsx"):
        normalize_path(_csv(tmp_path), tmp_path / "o.jsonl", sheet="x")
    with pytest.raises(SystemExit) as e:
        main(["normalize", "-i", str(tmp_path), "-o", str(tmp_path / "d"), "--sheet", "x"])
    assert e.value.code == 2
    monkeypatch.setitem(__import__("sys").modules, "openpyxl", None)
    with pytest.raises(SystemExit) as e:
        main(["normalize", "-i", str(src), "-o", str(tmp_path / "o.jsonl")])
    assert e.value.code == 2 and 'pip install "convmerge[xlsx]"' in capsys.readouterr().err


def _csv(tmp_path: Path) -> Path:
    path = tmp_path / "t.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    return path


def test_xlsx_cli_and_recipe(tmp_path: Path, capsys) -> None:
    from convmerge.cli import main
    from convmerge.recipe.schema import parse_recipe

    src = _workbook(tmp_path / "data.xlsx")
    main(["normalize", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--sheet", "second"])
    assert "1 records" in capsys.readouterr().err
    recipe = parse_recipe({"version": 1, "output": "o.jsonl", "sources": {"a": {
        "path": "data.xlsx", "normalize": {"sheet": "second"},
        "convert": {"from": "auto"}}}}, path=tmp_path / "r.yaml")  # fmt: skip
    assert recipe.sources["a"].sheet == "second"
