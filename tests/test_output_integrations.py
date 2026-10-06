"""Input, recipe commit and optional-format failures, independent of the CLI harness."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from convmerge import normalize_to_jsonl
from convmerge.mix import MixSource, mix_files
from convmerge.normalize.files import normalize_path
from convmerge.recipe import RecipeRunError, Step, engine


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_recipe_commit_failure_restores_previous_output(tmp_path, monkeypatch, kind):
    output = tmp_path / "result"
    if kind == "file":
        output.write_bytes(b"old")
    else:
        output.mkdir()
        (output / "old.jsonl").write_bytes(b"old")

    def run(stage):
        if kind == "file":
            stage.write_bytes(b"new")
        else:
            stage.mkdir()
            (stage / "new.jsonl").write_bytes(b"new")
        return {"records": 1}

    replace = os.replace

    def fail_publish(src, dst):
        if Path(dst) == output and not ("old" in str(src) or "previous" in str(src)):
            raise OSError("injected commit failure")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_publish)
    with pytest.raises((OSError, RecipeRunError), match="injected commit failure"):
        engine._execute(Step("a.normalize", "normalize", [], output, {}, run))
    assert (
        output.read_bytes() == b"old"
        if kind == "file"
        else (output / "old.jsonl").read_bytes() == b"old"
    )
    assert set(tmp_path.iterdir()) == {output}


def test_recipe_failed_restore_keeps_recovery_copy(tmp_path, monkeypatch):
    output = tmp_path / "result"
    output.mkdir()
    (output / "old.jsonl").write_bytes(b"old")
    replace = os.replace

    def fail_commit_and_restore(src, dst):
        if Path(dst) == output:
            raise OSError("injected storage failure")
        return replace(src, dst)

    def run(stage):
        stage.mkdir()
        (stage / "new.jsonl").write_bytes(b"new")
        return {}

    monkeypatch.setattr(os, "replace", fail_commit_and_restore)
    with pytest.raises(RecipeRunError, match="recovery copy") as exc:
        engine._execute(Step("a.normalize", "normalize", [], output, {}, run))
    copies = list(tmp_path.rglob("old.jsonl"))
    assert len(copies) == 1 and copies[0].read_bytes() == b"old"
    assert str(copies[0].parent) in str(exc.value)


def test_recipe_does_not_delete_unrelated_stage_names(tmp_path):
    output = tmp_path / "result.jsonl"
    part, old = tmp_path / ".result.jsonl.part", tmp_path / ".result.jsonl.old"
    part.write_bytes(b"unrelated partial")
    old.write_bytes(b"unrelated backup")

    def run(stage):
        stage.write_bytes(b"new")
        return {"records": 1}

    engine._execute(Step("a.convert", "convert", [], output, {}, run))
    assert part.read_bytes() == b"unrelated partial"
    assert old.read_bytes() == b"unrelated backup"


@pytest.mark.parametrize("kind", ["array", "single", "arrays_jsonl"])
def test_normalize_serialization_failure_preserves_output(tmp_path, monkeypatch, kind):
    import convmerge.normalize.jsonl as module

    rows = [{"x": 1}, {"x": 2}]
    text = json.dumps(rows) if kind == "array" else "".join(json.dumps(r) for r in rows)
    if kind == "arrays_jsonl":
        text = "\n".join(json.dumps([r]) for r in rows)
    src, dst = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    src.write_text(text, encoding="utf-8")
    dst.write_bytes(b"old")
    original = module._checked_dumps
    called = 0

    def failing_dumps(*args, **kwargs):
        nonlocal called
        called += 1
        if called == 2:
            raise OSError("injected serialization failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_checked_dumps", failing_dumps)
    with pytest.raises(OSError, match="injected serialization failure"):
        normalize_to_jsonl(src, dst)
    assert called == 2 and dst.read_bytes() == b"old"


def test_empty_normalization_successfully_replaces_output(tmp_path):
    src, dst = tmp_path / "empty.jsonl", tmp_path / "out.jsonl"
    src.write_bytes(b"")
    dst.write_bytes(b"old")
    assert normalize_to_jsonl(src, dst) == 0
    assert dst.read_bytes() == b""


def test_directory_cross_file_hardlink_is_rejected(tmp_path):
    from convmerge.io import SamePathError

    src, dst = tmp_path / "raw", tmp_path / "out"
    src.mkdir()
    dst.mkdir()
    a, b = src / "a.jsonl", src / "b.jsonl"
    a.write_bytes(b'{"a":1}\n')
    b.write_bytes(b'{"b":2}\n')
    os.link(b, dst / "a.jsonl")
    with pytest.raises(SamePathError):
        normalize_path(src, dst)
    assert b.read_bytes() == b'{"b":2}\n'
    assert not (dst / "b.jsonl").exists()


@pytest.mark.parametrize("format", ["parquet", "xlsx"])
def test_optional_input_iterator_failure_preserves_output(tmp_path, monkeypatch, format):
    dst = tmp_path / "out.jsonl"
    dst.write_bytes(b"old")
    if format == "xlsx":
        openpyxl = pytest.importorskip("openpyxl")
        from convmerge.normalize import tabular

        src = tmp_path / "in.xlsx"
        book = openpyxl.Workbook()
        book.active.append(["instruction", "output"])
        book.active.append(["q", "a"])
        book.save(src)
        book.close()

        def broken_rows(*args):
            yield ["instruction", "output"]
            yield ["q", "a"]
            raise OSError("injected input read failure")

        monkeypatch.setattr(tabular, "_xlsx_rows", broken_rows)
        call = tabular.xlsx_to_jsonl
    else:
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        from convmerge.normalize.parquet import parquet_to_jsonl

        src = tmp_path / "in.parquet"
        pq.write_table(pa.table({"instruction": ["q"], "output": ["a"]}), src)
        original = pq.ParquetFile

        class BrokenParquet:
            def __init__(self, path):
                self.file = original(path)

            def iter_batches(self, **kw):
                yield from self.file.iter_batches(**kw)
                raise OSError("injected input read failure")

            def close(self):
                self.file.close()

        monkeypatch.setattr(pq, "ParquetFile", BrokenParquet)
        call = parquet_to_jsonl
    with pytest.raises(OSError, match="injected input read failure"):
        call(src, dst)
    assert dst.read_bytes() == b"old"
    assert not list(tmp_path.glob(".convmerge-*.tmp"))


@pytest.mark.parametrize("sampler", ["v1", "v2"])
def test_mix_write_failure_preserves_existing_output(tmp_path, monkeypatch, sampler):
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text('{"x":1}\n{"x":2}\n', encoding="utf-8")
    dst.write_bytes(b"old")
    path_open, fdopen = Path.open, os.fdopen

    class FailingWriter:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, text):
            self.stream.write(text)
            raise OSError("injected disk full")

        def writelines(self, lines):
            for line in lines:
                self.write(line)

    def open_path(path, mode="r", *args, **kwargs):
        stream = path_open(path, mode, *args, **kwargs)
        return FailingWriter(stream) if path == dst and "w" in mode else stream

    monkeypatch.setattr(Path, "open", open_path)
    monkeypatch.setattr(os, "fdopen", lambda *a, **k: FailingWriter(fdopen(*a, **k)))
    with pytest.raises(OSError, match="injected disk full"):
        mix_files([MixSource(src, 1)], dst, sampler=sampler)
    assert dst.read_bytes() == b"old"


def test_failed_recipe_normalize_does_not_mark_lock_success(tmp_path):
    from convmerge.recipe import parse_recipe, run

    source = tmp_path / "source.jsonl"
    source.write_text('{"instruction":"q","output":"a"}\n', encoding="utf-8")
    recipe = parse_recipe(
        {
            "output": "train.jsonl",
            "sources": {"a": {"path": "source.jsonl", "convert": {"from": "auto"}}},
        },
        path=tmp_path / "recipe.json",
    )
    run(recipe, log=lambda text: None)
    previous_lock = recipe.lock_path.read_bytes()
    previous_output = recipe.output.read_bytes()
    source.write_text('{"instruction":"q","output":"a"}\n{broken\n', encoding="utf-8")
    with pytest.raises(RecipeRunError, match="normalize"):
        run(recipe, log=lambda text: None)
    assert recipe.lock_path.read_bytes() == previous_lock
    assert recipe.output.read_bytes() == previous_output


def test_folder_cli_returns_failure_after_independent_success(tmp_path):
    from convmerge.cli import main

    source, output = tmp_path / "raw", tmp_path / "out"
    source.mkdir()
    output.mkdir()
    (source / "a.jsonl").write_bytes(b'{"x":1}\n')
    (source / "b.jsonl").write_bytes(b'{"x":1}\n{broken\n')
    (output / "b.jsonl").write_bytes(b"old")
    with pytest.raises(SystemExit) as exc:
        main(["normalize", "-i", str(source), "-o", str(output)])
    assert exc.value.code == 1
    assert json.loads((output / "a.jsonl").read_text()) == {"x": 1}
    assert (output / "b.jsonl").read_bytes() == b"old"


@pytest.mark.parametrize("kind", ["csv", "tsv", "xlsx", "parquet"])
def test_direct_format_api_protects_source(tmp_path, kind):
    from convmerge.io import SamePathError
    from convmerge.normalize.tabular import table_to_jsonl, xlsx_to_jsonl

    source = tmp_path / f"source.{kind}"
    source.write_bytes(b"original file")
    if kind == "parquet":
        pytest.importorskip("pyarrow")
        from convmerge.normalize.parquet import parquet_to_jsonl

        call = parquet_to_jsonl
    elif kind == "xlsx":
        call = xlsx_to_jsonl
    else:
        call = table_to_jsonl
    with pytest.raises(SamePathError):
        call(source, source)
    assert source.read_bytes() == b"original file"
