"""Deterministic train/test split for JSONL files.

Pure Python implementation (no numpy dependency): memory use is dominated by
a set of up to ``max_samples`` line indices. For typical SFT dataset sizes
(<=1e7 rows) this is fine; if you have much larger corpora, prefer sharding
+ per-shard splits or supply your own sampler.
"""

from __future__ import annotations

import random
from pathlib import Path


def train_test_split(
    src: str | Path,
    out_dir: str | Path,
    *,
    train_ratio: float = 0.98,
    seed: int = 42,
    max_samples: int | None = None,
    train_name: str = "train.jsonl",
    test_name: str = "test.jsonl",
) -> tuple[Path, Path, int, int]:
    """Shuffle-split a JSONL file into ``train_name`` and ``test_name``.

    Two passes over the file: the first counts non-blank lines, the second
    streams them back and routes each selected line to the train or test
    writer. ``max_samples`` sub-samples line indices before the split, which
    is equivalent to shuffling and truncating the whole dataset.

    ``train_ratio`` must satisfy ``0.0 < train_ratio < 1.0``.

    Returns ``(train_path, test_path, n_train_written, n_test_written)``.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio!r}")

    src_p = Path(src)
    if not src_p.is_file():
        raise FileNotFoundError(f"No such file: {src_p}")

    with src_p.open(encoding="utf-8") as rf:
        n_lines = sum(1 for line in rf if line.strip())
    if n_lines == 0:
        raise RuntimeError(f"No samples found in {src_p}")

    rng = random.Random(seed)
    if max_samples is not None and max_samples < n_lines:
        m = max_samples
        chosen_indices = rng.sample(range(n_lines), m)
    else:
        m = n_lines
        chosen_indices = list(range(n_lines))

    rng.shuffle(chosen_indices)
    n_train = int(m * train_ratio)
    train_set = set(chosen_indices[:n_train])
    test_set = set(chosen_indices[n_train:])

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    train_path = out_p / train_name
    test_path = out_p / test_name

    ti = 0
    written_train = 0
    written_test = 0
    with (
        src_p.open(encoding="utf-8") as rf,
        train_path.open("w", encoding="utf-8") as wtr,
        test_path.open("w", encoding="utf-8") as wte,
    ):
        for line in rf:
            if not line.strip():
                continue
            if ti in train_set:
                wtr.write(line if line.endswith("\n") else line + "\n")
                written_train += 1
            elif ti in test_set:
                wte.write(line if line.endswith("\n") else line + "\n")
                written_test += 1
            ti += 1

    return train_path, test_path, written_train, written_test
