"""Streaming parquet-to-JSONL conversion.

Requires the optional ``parquet`` extra::

    pip install "convmerge[parquet]"  # or "convmerge[all]"
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

from convmerge._output import atomic_text_writer
from convmerge.io import refuse_overwrite


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
            "Install with: pip install 'convmerge[parquet]' (or [all])"
        ) from e

    refuse_overwrite([src], [dst])
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with atomic_text_writer(dst_p) as wf, closing(pq.ParquetFile(src_p)) as pf:
        for batch in pf.iter_batches(batch_size=batch_rows):
            for row in batch.to_pylist():
                wf.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                n_written += 1
    return n_written
