"""Tests for convmerge.normalize.parquet (requires pyarrow)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from convmerge.normalize.parquet import parquet_to_jsonl  # noqa: E402


def test_parquet_roundtrip(tmp_path: Path) -> None:
    table = pa.table({"id": [1, 2, 3], "text": ["a", "b", "c"]})
    src = tmp_path / "in.parquet"
    pq.write_table(table, src)

    dst = tmp_path / "out.jsonl"
    n = parquet_to_jsonl(src, dst, batch_rows=2)
    assert n == 3
    rows = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"id": 1, "text": "a"},
        {"id": 2, "text": "b"},
        {"id": 3, "text": "c"},
    ]
