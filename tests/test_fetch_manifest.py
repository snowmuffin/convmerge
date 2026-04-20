"""Tests for YAML manifest parsing and entry classification."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

pytest.importorskip("yaml")

from convmerge.fetch.manifest import (  # noqa: E402
    DatasetEntry,
    classify_entry,
    load_manifest,
    sanitize_name,
)


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "m.yaml"
    p.write_text(dedent(text), encoding="utf-8")
    return p


def test_sanitize_name_strips_bad_chars() -> None:
    assert sanitize_name("foo/bar:baz") == "foo_bar_baz"
    assert sanitize_name("   ") == "unnamed"
    assert sanitize_name("a b\tc") == "a_b_c"


def test_load_manifest_basic(tmp_path: Path) -> None:
    m = load_manifest(
        _write(
            tmp_path,
            """
            version: 1
            defaults:
              output_root: ./raw
              on_error: continue
              resume: true
            auth:
              hf_token_env: HF_TOKEN
              github_token_env: GITHUB_TOKEN
            datasets:
              - name: ds-a
                hf: org/dataset-a
                split: train
              - name: ds-b
                url: https://github.com/org/repo
                ext: [.jsonl]
            """,
        )
    )
    assert m.version == 1
    assert m.defaults.output_root == "./raw"
    assert m.defaults.on_error == "continue"
    assert m.auth.hf.env == "HF_TOKEN"
    assert m.auth.github.env == "GITHUB_TOKEN"
    assert [d.name for d in m.datasets] == ["ds-a", "ds-b"]
    assert m.datasets[0].hf == "org/dataset-a"
    assert m.datasets[1].ext == (".jsonl",)


def test_load_manifest_rejects_both_hf_and_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_manifest(
            _write(
                tmp_path,
                """
                version: 1
                datasets:
                  - name: bad
                    hf: org/ds
                    url: https://github.com/foo/bar
                """,
            )
        )


def test_load_manifest_rejects_neither_hf_nor_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_manifest(
            _write(
                tmp_path,
                """
                version: 1
                datasets:
                  - name: bad
                """,
            )
        )


def test_load_manifest_bad_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_manifest(
            _write(
                tmp_path,
                """
                version: 99
                datasets: []
                """,
            )
        )


def test_classify_entry_hf() -> None:
    assert classify_entry(DatasetEntry(name="x", hf="org/ds")) == "hf"


def test_classify_entry_raw_github() -> None:
    e = DatasetEntry(name="x", url="https://raw.githubusercontent.com/o/r/main/a.jsonl")
    assert classify_entry(e) == "url_raw"


def test_classify_entry_raw_suffix() -> None:
    e = DatasetEntry(name="x", url="https://example.com/foo/bar.jsonl")
    assert classify_entry(e) == "url_raw"


def test_classify_entry_github_tree_default() -> None:
    e = DatasetEntry(name="x", url="https://github.com/owner/repo")
    assert classify_entry(e) == "url_github_tree"


def test_classify_entry_github_clone() -> None:
    e = DatasetEntry(name="x", url="https://github.com/owner/repo", mode="clone")
    assert classify_entry(e) == "url_github_clone"


def test_classify_entry_rejects_unknown_host() -> None:
    with pytest.raises(ValueError):
        classify_entry(DatasetEntry(name="x", url="https://gitlab.com/o/r"))
