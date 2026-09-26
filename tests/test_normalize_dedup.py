"""Tests for convmerge.normalize.dedup."""

from __future__ import annotations

import json
import tempfile
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


# --- issue #14: bounded-memory seen store ---------------------------------


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_dedup_sqlite_matches_memory(tmp_path: Path) -> None:
    data = [{"a": 1}, {"a": 1}, {"a": 2}, {"a": 1, "b": 2}, {"a": 2}]
    src = _write(tmp_path / "src.jsonl", data)

    mem_out = tmp_path / "mem.jsonl"
    sql_out = tmp_path / "sql.jsonl"
    mem = deduplicate_jsonl(src, mem_out, seen_store="memory")
    sql = deduplicate_jsonl(src, sql_out, seen_store="sqlite")

    assert mem == sql == (5, 3)
    assert _rows(mem_out) == _rows(sql_out)


def test_dedup_sqlite_explicit_db_path(tmp_path: Path) -> None:
    src = _write(tmp_path / "src.jsonl", [{"a": 1}, {"a": 1}, {"a": 2}])
    dst = tmp_path / "out.jsonl"
    db = tmp_path / "seen.sqlite"
    total, kept = deduplicate_jsonl(src, dst, seen_store="sqlite", seen_db=db)
    assert (total, kept) == (3, 2)
    # An explicit db path is left in place (not auto-removed).
    assert db.is_file()


def test_dedup_temp_sqlite_is_cleaned_up(tmp_path: Path) -> None:
    src = _write(tmp_path / "src.jsonl", [{"a": 1}, {"a": 1}])
    dst = tmp_path / "out.jsonl"
    before = set(Path(tempfile.gettempdir()).glob("convmerge-dedupe-*.sqlite"))
    deduplicate_jsonl(src, dst, seen_store="sqlite")
    after = set(Path(tempfile.gettempdir()).glob("convmerge-dedupe-*.sqlite"))
    assert after == before  # temp db removed on completion


def test_dedup_unknown_seen_store(tmp_path: Path) -> None:
    src = _write(tmp_path / "src.jsonl", [{"a": 1}])
    dst = tmp_path / "out.jsonl"
    with pytest.raises(ValueError):
        deduplicate_jsonl(src, dst, seen_store="redis")


def test_dedupe_stats_separates_invalid_from_duplicates(tmp_path: Path) -> None:
    from convmerge.normalize.dedup import DedupeStats

    src = tmp_path / "in.jsonl"
    src.write_text('{"a": 1}\n{"a": 1}\nBAD\n{"a": 2}\n', encoding="utf-8")
    stats = DedupeStats()
    total, kept = deduplicate_jsonl(src, tmp_path / "out.jsonl", stats=stats)
    assert (total, kept) == (4, 2)
    assert stats.duplicates == 1
    assert stats.invalid_json == 1
    assert stats.first_invalid_line == 3
