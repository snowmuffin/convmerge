"""Tests for convmerge.normalize.dedup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.normalize.dedup import deduplicate_jsonl


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def test_dedup_full_record(tmp_path: Path) -> None:
    src = _write(
        tmp_path / "src.jsonl",
        [{"a": 1}, {"a": 1}, {"a": 2}, {"a": 1, "b": 2}],
    )
    dst = tmp_path / "out.jsonl"
    total, kept = deduplicate_jsonl(src, dst)
    assert (total, kept) == (4, 3)


def test_dedup_by_keys(tmp_path: Path) -> None:
    src = _write(
        tmp_path / "src.jsonl",
        [
            {"id": 1, "q": "same"},
            {"id": 2, "q": "same"},
            {"id": 3, "q": "different"},
        ],
    )
    dst = tmp_path / "out.jsonl"
    total, kept = deduplicate_jsonl(src, dst, keys=["q"])
    assert (total, kept) == (3, 2)


def test_dedup_sha256(tmp_path: Path) -> None:
    src = _write(tmp_path / "src.jsonl", [{"a": 1}, {"a": 1}])
    dst = tmp_path / "out.jsonl"
    total, kept = deduplicate_jsonl(src, dst, algorithm="sha256")
    assert (total, kept) == (2, 1)


def test_dedup_unknown_algorithm(tmp_path: Path) -> None:
    src = _write(tmp_path / "src.jsonl", [{"a": 1}])
    dst = tmp_path / "out.jsonl"
    with pytest.raises(ValueError):
        deduplicate_jsonl(src, dst, algorithm="blake2zz")
