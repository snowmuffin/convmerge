"""Regression contracts for safe file publication (no network or model needed)."""

from __future__ import annotations

import json
import os

import pytest

from convmerge import convert_file, normalize_to_jsonl
from convmerge.cli import main
from convmerge.io import SamePathError
from convmerge.normalize.files import normalize_path
from convmerge.normalize.tabular import table_to_jsonl


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("entry", ["api", "cli"])
def test_normalize_failure_preserves_output(tmp_path, existing, entry):
    src, dst = tmp_path / "input.jsonl", tmp_path / "out.jsonl"
    src.write_text('{"x":1}\n{broken\n', encoding="utf-8")
    old = b'{"messages": [{"role": "user", "content": "old"}]}\n'
    if existing:
        dst.write_bytes(old)
    if entry == "api":
        with pytest.raises(ValueError):
            normalize_to_jsonl(src, dst)
    else:
        with pytest.raises(SystemExit) as exc:
            main(["normalize", "-i", str(src), "-o", str(dst)])
        assert exc.value.code == 1
    assert dst.read_bytes() == old if existing else not dst.exists()
    assert set(tmp_path.iterdir()) == ({src, dst} if existing else {src})


@pytest.mark.parametrize("kind", ["same", "symlink", "hardlink"])
def test_direct_normalize_refuses_input_alias(tmp_path, kind):
    src = tmp_path / "input.jsonl"
    src.write_text('{"x":1}\n{"x":2}\n', encoding="utf-8")
    dst = src if kind == "same" else tmp_path / "alias.jsonl"
    if kind == "symlink":
        try:
            dst.symlink_to(src)
        except OSError as exc:
            pytest.skip(f"symlinks not available: {exc}")
    elif kind == "hardlink":
        os.link(src, dst)
    before = src.read_bytes()
    with pytest.raises(SamePathError):
        normalize_to_jsonl(src, dst)
    assert src.read_bytes() == before


@pytest.mark.parametrize("existing", [False, True])
def test_csv_error_after_valid_row_preserves_output(tmp_path, existing):
    src, dst = tmp_path / "in.csv", tmp_path / "out.jsonl"
    src.write_text('instruction,output\nq,a\n"unfinished', encoding="utf-8")
    if existing:
        dst.write_bytes(b"previous\n")
    with pytest.raises(ValueError):
        table_to_jsonl(src, dst)
    assert dst.read_bytes() == b"previous\n" if existing else not dst.exists()


def test_directory_collision_refused_before_any_writes(tmp_path):
    src, dst = tmp_path / "raw", tmp_path / "out"
    src.mkdir()
    (src / "a.json").write_text('[{"x":1}]', encoding="utf-8")
    (src / "a.jsonl").write_text('{"x":2}\n', encoding="utf-8")
    (src / "0.jsonl").write_text('{"x":0}\n', encoding="utf-8")
    with pytest.raises(SamePathError):
        normalize_path(src, dst)
    assert not dst.exists()


def test_nested_output_directory_refused(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    (src / "a.jsonl").write_text('{"x":1}\n', encoding="utf-8")
    with pytest.raises(SamePathError):
        normalize_path(src, src / "normalized")
    assert not (src / "normalized").exists()


def test_directory_preserves_only_failed_file(tmp_path):
    src, dst = tmp_path / "raw", tmp_path / "out"
    src.mkdir()
    dst.mkdir()
    (src / "bad.jsonl").write_text('{"x":1}\n{broken\n', encoding="utf-8")
    (src / "good.jsonl").write_text('{"x":2}\n', encoding="utf-8")
    (dst / "bad.jsonl").write_bytes(b"previous\n")
    result = normalize_path(src, dst)
    assert len(result.failed) == len(result.files) == 1
    assert (dst / "bad.jsonl").read_bytes() == b"previous\n"
    assert json.loads((dst / "good.jsonl").read_text()) == {"x": 2}


@pytest.mark.parametrize("existing", [False, True])
def test_convert_final_path_not_changed_while_running(tmp_path, monkeypatch, existing):
    import convmerge.convert as module

    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text(json.dumps({"instruction": "q", "output": "a"}) + "\n", encoding="utf-8")
    if existing:
        dst.write_bytes(b"previous\n")
    original = module._process

    def inspect_write(*args, **kwargs):
        assert dst.read_bytes() == b"previous\n" if existing else not dst.exists()
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_process", inspect_write)
    convert_file(src, dst, adapter_name="auto", output_format="messages")
    assert json.loads(dst.read_text(encoding="utf-8"))["messages"][-1]["content"] == "a"
