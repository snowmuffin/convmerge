"""Tests for convmerge.normalize.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.normalize.jsonl import (
    detect_jsonl_shape,
    iter_json_records,
    load_jsonl,
    normalize_to_jsonl,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_detect_shape_jsonl(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.jsonl", '{"a":1}\n{"a":2}\n')
    assert detect_jsonl_shape(p) == "jsonl"


def test_detect_shape_json_array(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.json", '[{"a":1},{"a":2}]')
    assert detect_jsonl_shape(p) == "json_array"


def test_detect_shape_single_line_concatenated(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.jsonl", '{"a":1}{"a":2}{"a":3}')
    assert detect_jsonl_shape(p) == "single_line"


def test_detect_shape_single_object(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.jsonl", '{"a":1}')
    assert detect_jsonl_shape(p) == "single_line"


def test_detect_shape_empty(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.jsonl", "")
    assert detect_jsonl_shape(p) == "empty"


def test_normalize_json_array(tmp_path: Path) -> None:
    src = _write(tmp_path / "a.json", '[{"x":1},{"x":2}]')
    dst = tmp_path / "out.jsonl"
    n = normalize_to_jsonl(src, dst)
    assert n == 2
    lines = dst.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in lines] == [{"x": 1}, {"x": 2}]


def test_normalize_single_line_concatenated(tmp_path: Path) -> None:
    src = _write(tmp_path / "a.jsonl", '{"x":1}{"x":2}{"x":3}')
    dst = tmp_path / "out.jsonl"
    n = normalize_to_jsonl(src, dst)
    assert n == 3
    assert dst.read_text(encoding="utf-8").count("\n") == 3


def test_normalize_valid_jsonl_passthrough(tmp_path: Path) -> None:
    src = _write(tmp_path / "a.jsonl", '{"x":1}\n\n{"x":2}\n')
    dst = tmp_path / "out.jsonl"
    n = normalize_to_jsonl(src, dst)
    assert n == 2


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.jsonl", '{"a":1}\n\n{"a":2}\n')
    rows = load_jsonl(p)
    assert rows == [{"a": 1}, {"a": 2}]


def test_iter_json_records_handles_json_single_object(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.json", '{"only":"one"}')
    assert list(iter_json_records(p)) == [{"only": "one"}]


def test_iter_json_records_max_rows(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.jsonl", '{"x":1}\n{"x":2}\n{"x":3}\n')
    assert list(iter_json_records(p, max_rows=2)) == [{"x": 1}, {"x": 2}]


def test_normalize_jsonl_sanitizes_bom_crlf_and_trailing_whitespace(tmp_path: Path) -> None:
    src = _write(tmp_path / "messy.jsonl", '\ufeff{"x":1}   \r\n{"x":2}\t \r\n')
    dst = tmp_path / "out.jsonl"

    n = normalize_to_jsonl(src, dst)

    assert n == 2
    assert dst.read_text(encoding="utf-8") == '{"x":1}\n{"x":2}\n'


def test_normalize_jsonl_rejects_trailing_comma_with_file_and_line(tmp_path: Path) -> None:
    src = _write(tmp_path / "bad.jsonl", '{"x":1},\n{"x":2}\n')
    dst = tmp_path / "out.jsonl"

    with pytest.raises(ValueError) as exc_info:
        normalize_to_jsonl(src, dst)

    message = str(exc_info.value)
    assert "bad.jsonl" in message
    assert "line 1" in message
    assert "trailing comma" in message
