"""CSV and TSV files → JSONL, one object per row keyed by the header.

Spreadsheet exports are a common way instruction data arrives (``instruction,
input,output`` columns). The first row is the header; every later row becomes
an object of strings. Cells that span lines (quoted) are kept whole, a UTF-8
byte order mark is ignored, empty rows are skipped, and header cells that are
empty or repeated get unique names (``column_3``, ``output_2``) so no value is
lost. Values stay strings; ``convert`` reads JSON-looking strings such as a
serialized ``messages`` column itself.
"""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

TABLE_EXTENSIONS: tuple[str, ...] = (".csv", ".tsv")


def table_to_jsonl(src: str | Path, dst: str | Path, *, encoding: str = "utf-8") -> int:
    """Rewrite a ``.csv`` (comma) or ``.tsv`` (tab) file as JSONL; return rows written."""
    src_p, dst_p = Path(src), Path(dst)
    delimiter = "\t" if src_p.suffix.lower() == ".tsv" else ","
    _allow_long_fields()
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    enc = "utf-8-sig" if encoding.lower().replace("_", "-") in ("utf-8", "utf8") else encoding
    with src_p.open(encoding=enc, newline="") as fin, dst_p.open("w", encoding="utf-8") as fout:
        reader = csv.reader(fin, delimiter=delimiter, strict=True)
        try:
            written = _write_rows(reader, fout)
        except csv.Error as e:
            raise ValueError(f"{src_p}: line {reader.line_num}: {e}") from None
    return written


def _write_rows(reader: Iterator[list[str]], fout: TextIO) -> int:
    written = 0
    header = next(reader, None)
    if header is None:
        return 0
    names = _column_names(header)
    for row in reader:
        if not any(cell.strip() for cell in row):
            continue
        record = {names[i] if i < len(names) else f"column_{i + 1}": cell
                  for i, cell in enumerate(row)}  # fmt: skip
        for name in names[len(row) :]:
            record[name] = ""
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
    return written


def _column_names(header: list[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for i, cell in enumerate(header):
        base = cell.strip() or f"column_{i + 1}"
        name, n = base, 1
        while name in seen:
            n += 1
            name = f"{base}_{n}"
        seen.add(name)
        names.append(name)
    return names


def _allow_long_fields() -> None:
    # The csv module rejects fields over 128 KiB by default; long answers exceed it.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10
