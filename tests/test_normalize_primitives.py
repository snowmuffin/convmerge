"""Tests for the directory-level primitives added to convmerge.normalize."""

from __future__ import annotations

import json
from pathlib import Path

from convmerge.normalize import (
    head_preview,
    head_preview_dir,
    normalize_dir,
    prune_by_suffix,
    scan_shapes,
)
from convmerge.normalize.turns import filter_by_min_turns


def test_normalize_dir_walks_json_and_jsonl(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.json").write_text('[{"x":1},{"x":2}]')
    (src / "sub").mkdir()
    (src / "sub" / "b.jsonl").write_text('{"y":1}{"y":2}{"y":3}')
    dst = tmp_path / "dst"

    n_files, n_rows = normalize_dir(src, dst)

    assert n_files == 2
    assert n_rows == 5
    assert (dst / "a.jsonl").read_text().count("\n") == 2
    assert (dst / "sub" / "b.jsonl").read_text().count("\n") == 3


def test_normalize_dir_skips_non_matching(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "README.md").write_text("skip me")
    (src / "a.jsonl").write_text('{"x":1}\n')
    dst = tmp_path / "dst"

    n_files, n_rows = normalize_dir(src, dst)
    assert n_files == 1
    assert n_rows == 1
    assert not (dst / "README.md").exists()


def test_prune_by_suffix_removes_matches(tmp_path: Path) -> None:
    (tmp_path / "a-res.jsonl").write_text("x")
    (tmp_path / "b.jsonl").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c-res.jsonl").write_text("x")

    removed = prune_by_suffix(tmp_path, ["-res.jsonl"])
    names = {p.name for p in removed}
    assert names == {"a-res.jsonl", "c-res.jsonl"}
    assert not (tmp_path / "a-res.jsonl").exists()
    assert (tmp_path / "b.jsonl").exists()


def test_prune_by_suffix_dry_run(tmp_path: Path) -> None:
    (tmp_path / "a-res.jsonl").write_text("x")
    removed = prune_by_suffix(tmp_path, ["-res.jsonl"], dry_run=True)
    assert len(removed) == 1
    assert (tmp_path / "a-res.jsonl").exists()


def test_head_preview_returns_first_line(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text('{"x":1}\n{"x":2}\n')
    assert head_preview(p) == '{"x":1}'


def test_head_preview_dir_filters_by_extension(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text('{"x":1}\n')
    (tmp_path / "b.md").write_text("not me")
    previews = head_preview_dir(tmp_path, extensions=[".jsonl"])
    assert len(previews) == 1
    assert list(previews.values())[0] == '{"x":1}'


def test_scan_shapes_classifies_each_file(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text('{"x":1}\n{"x":2}\n')
    (tmp_path / "b.json").write_text('[{"x":1}]')
    (tmp_path / "c.jsonl").write_text('{"x":1}{"x":2}')
    shapes = scan_shapes(tmp_path)
    mapped = {p.name: s for p, s in shapes.items()}
    assert mapped == {"a.jsonl": "jsonl", "b.json": "json_array", "c.jsonl": "single_line"}


def test_filter_by_min_turns_drops_short_rows(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    dst = tmp_path / "dst.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    total, kept = filter_by_min_turns(src, dst, min_turns=1)
    assert total == 2
    assert kept == 1
    assert dst.read_text().strip().count("\n") == 0
