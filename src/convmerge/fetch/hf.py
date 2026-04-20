"""HuggingFace fetcher.

A deliberately thin wrapper around ``datasets.load_dataset(...).to_json(...)``.
We never reimplement dataset loading; if you only need one dataset, call
``datasets.load_dataset`` directly in your own code.
"""

from __future__ import annotations

from pathlib import Path


def download_hf_dataset(
    dataset_id: str,
    dst_path: str | Path,
    *,
    config: str | None = None,
    split: str | None = None,
    token: str | None = None,
) -> Path:
    """Load a HuggingFace dataset and dump it to a JSONL file.

    ``config`` and ``split`` default to ``None`` / ``"train"``, which matches
    the shape of most SFT datasets. Raises ``ImportError`` with an install hint
    when the ``datasets`` package is missing.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace entries require the 'datasets' package. "
            "Install with: pip install 'convmerge[fetch-hf]'"
        ) from e

    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, object] = {}
    if config is not None:
        load_kwargs["name"] = config
    load_kwargs["split"] = split or "train"
    if token:
        load_kwargs["token"] = token

    ds = load_dataset(dataset_id, **load_kwargs)
    ds.to_json(str(dst))
    return dst
