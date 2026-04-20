"""Streaming parquet-to-JSONL conversion.

Requires the optional ``parquet`` extra::

    pip install "convmerge[parquet]"
"""

from __future__ import annotations

import json
from pathlib import Path


def parquet_to_jsonl(src: str | Path, dst: str | Path, *, batch_rows: int = 65536) -> int:
    """Stream a parquet file into a JSONL file row by row.

    Uses PyArrow's record-batch iterator so the whole table never needs to live
    in memory. Returns the number of rows written.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "pyarrow is required for parquet conversion. "
            "Install with: pip install 'convmerge[parquet]'"
        ) from e

    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(src_p)
    n_written = 0
    with dst_p.open("w", encoding="utf-8") as wf:
        for batch in pf.iter_batches(batch_size=batch_rows):
            for row in batch.to_pylist():
                wf.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                n_written += 1
    return n_written
