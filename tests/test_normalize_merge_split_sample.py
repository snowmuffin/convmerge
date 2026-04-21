"""Tests for merge, split, sample, and resume helpers."""

from __future__ import annotations

import json
from pathlib import Path

from convmerge.normalize.merge import collect_jsonl_tree, merge_jsonl
from convmerge.normalize.resume import count_lines, trim_corrupt_tail, truncate_to_n_lines
from convmerge.normalize.sample import reservoir_sample, sample_random
from convmerge.normalize.split import train_test_split


def test_merge_jsonl_concatenates_and_skips_blank(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"x":1}\n\n{"x":2}\n')
    b.write_text('{"y":1}\n')
    out = tmp_path / "merged.jsonl"
    n = merge_jsonl([a, b], out)
    assert n == 3
    assert out.read_text().count("\n") == 3


def test_merge_jsonl_validate_drops_junk(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    a.write_text('{"x":1}\nnot json\n{"x":2}\n')
    out = tmp_path / "merged.jsonl"
    n = merge_jsonl([a], out, validate=True)
    assert n == 2


def test_merge_jsonl_skips_missing_sources(tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    real.write_text('{"x":1}\n')
    missing = tmp_path / "missing.jsonl"
    out = tmp_path / "merged.jsonl"
    n = merge_jsonl([missing, real], out)
    assert n == 1


def test_collect_jsonl_tree_walks_nested(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    (root / "a" / "1.jsonl").write_text('{"x":1}\n{"bad":\n')  # second line invalid
    (root / "b.jsonl").write_text('{"y":1}\n{"y":2}\n')
    out = tmp_path / "all.jsonl"
    written, skipped = collect_jsonl_tree([root], out)
    assert written == 3
    assert skipped == 1


def test_train_test_split_deterministic(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    src.write_text("\n".join(json.dumps({"i": i}) for i in range(100)) + "\n")
    tr, te, n_tr, n_te = train_test_split(src, tmp_path / "out", train_ratio=0.8, seed=42)
    assert n_tr + n_te == 100
    assert abs(n_tr - 80) <= 1
    tr2, te2, _, _ = train_test_split(src, tmp_path / "out2", train_ratio=0.8, seed=42)
    assert tr.read_text() == tr2.read_text()
    assert te.read_text() == te2.read_text()


def test_train_test_split_with_max_samples(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    src.write_text("\n".join(json.dumps({"i": i}) for i in range(100)) + "\n")
    _, _, n_tr, n_te = train_test_split(
        src, tmp_path / "out", train_ratio=0.8, seed=1, max_samples=20
    )
    assert n_tr + n_te == 20


def test_sample_random_returns_k_lines(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    src.write_text("\n".join(str(i) for i in range(50)) + "\n")
    dst = tmp_path / "s.jsonl"
    written, total = sample_random(src, dst, k=10, seed=3)
    assert total == 50
    assert written == 10
    assert dst.read_text().count("\n") == 10


def test_sample_random_k_larger_than_population(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    src.write_text("a\nb\n")
    dst = tmp_path / "s.jsonl"
    written, total = sample_random(src, dst, k=10)
    assert total == 2
    assert written == 2


def test_reservoir_sample_bounded(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    src.write_text("\n".join(str(i) for i in range(1000)) + "\n")
    dst = tmp_path / "s.jsonl"
    written, seen = reservoir_sample(src, dst, k=25, seed=9)
    assert written == 25
    assert seen == 1000


def test_trim_corrupt_tail_and_companions(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.jsonl"
    split1 = tmp_path / "split1.jsonl"
    split2 = tmp_path / "split2.jsonl"
    bundle.write_text('{"ok":1}\n{"ok":2}\n{broken')
    split1.write_text("a\nb\nc\n")
    split2.write_text("a\nb\nc\n")

    n = trim_corrupt_tail(bundle, companions=(split1, split2))
    assert n == 2
    assert count_lines(bundle) == 2
    assert count_lines(split1) == 2
    assert count_lines(split2) == 2


def test_truncate_to_n_lines(tmp_path: Path) -> None:
    p = tmp_path / "p.jsonl"
    p.write_text("a\nb\nc\nd\n")
    truncate_to_n_lines(p, 2)
    assert p.read_text() == "a\nb\n"
    truncate_to_n_lines(p, 0)
    assert p.read_text() == ""


def test_count_lines_missing_file(tmp_path: Path) -> None:
    assert count_lines(tmp_path / "nope.jsonl") == 0
