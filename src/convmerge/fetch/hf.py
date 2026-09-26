"""HuggingFace fetcher.

A deliberately thin wrapper around ``datasets.load_dataset(...).to_json(...)``.
We never reimplement dataset loading; if you only need one dataset, call
``datasets.load_dataset`` directly in your own code.
"""

from __future__ import annotations

import json
from pathlib import Path


def download_hf_dataset(
    dataset_id: str,
    dst_path: str | Path,
    *,
    config: str | None = None,
    split: str | None = None,
    token: str | None = None,
    max_rows: int | None = None,
) -> Path:
    """Load a HuggingFace dataset and dump it to a JSONL file.

    ``config`` and ``split`` default to ``None`` / ``"train"``, which matches
    the shape of most SFT datasets. Raises ``ImportError`` with an install hint
    when the ``datasets`` package is missing.

    With ``max_rows`` the split is opened in streaming mode and only the first
    N rows are written, so nothing else is downloaded. Values JSON cannot
    represent (e.g. decoded images) are written as strings.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace entries require the 'datasets' package. "
            "Install with: pip install 'convmerge[fetch-all]' (or [fetch-hf] or [all])"
        ) from e

    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, object] = {}
    if config is not None:
        load_kwargs["name"] = config
    load_kwargs["split"] = split or "train"
    if token:
        load_kwargs["token"] = token

    if max_rows is None:
        ds = load_dataset(dataset_id, **load_kwargs)
        ds.to_json(str(dst))
        return dst

    stream = load_dataset(dataset_id, streaming=True, **load_kwargs)
    tmp = dst.with_name(f".{dst.name}.part")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for row in stream.take(max_rows):
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst
