"""End-to-end runner tests (all backends mocked)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from convmerge.fetch import runner  # noqa: E402
from convmerge.fetch.manifest import (  # noqa: E402
    AuthConfig,
    DatasetEntry,
    Defaults,
    Manifest,
    TokenSpec,
)


def _make_manifest(entries: list[DatasetEntry], root: Path) -> Manifest:
    return Manifest(
        version=1,
        auth=AuthConfig(hf=TokenSpec(env="HF_X"), github=TokenSpec(env="GH_X")),
        defaults=Defaults(output_root=str(root), on_error="continue", resume=True),
        datasets=tuple(entries),
    )


def test_runner_dispatches_all_backends(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    def fake_hf(dataset_id, dst, *, config=None, split=None, token=None, max_rows=None):
        calls.append(("hf", {"id": dataset_id, "dst": str(dst), "token": token}))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("{}\n", encoding="utf-8")
        return Path(dst)

    def fake_raw(url, dst, *, token=None, max_rows=None):
        calls.append(("raw", {"url": url, "dst": str(dst), "token": token}))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("{}\n", encoding="utf-8")
        return Path(dst)

    def fake_tree(url, dst, *, ext=(), token=None, max_rows=None):
        calls.append(("tree", {"url": url, "dst": str(dst), "ext": ext, "token": token}))
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "a.jsonl").write_text("{}\n", encoding="utf-8")
        return [Path(dst) / "a.jsonl"]

    def fake_clone(url, dst, *, token=None, lfs=False):
        calls.append(("clone", {"url": url, "dst": str(dst), "token": token, "lfs": lfs}))
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "file.bin").write_text("x", encoding="utf-8")
        return Path(dst)

    import convmerge.fetch.git as gitmod
    import convmerge.fetch.github as ghmod
    import convmerge.fetch.hf as hfmod

    monkeypatch.setattr(hfmod, "download_hf_dataset", fake_hf)
    monkeypatch.setattr(ghmod, "download_raw_file", fake_raw)
    monkeypatch.setattr(ghmod, "fetch_repo_tree_files", fake_tree)
    monkeypatch.setattr(gitmod, "clone_repo", fake_clone)

    monkeypatch.setenv("HF_X", "HF_TOKEN_VALUE")
    monkeypatch.setenv("GH_X", "GH_TOKEN_VALUE")

    entries = [
        DatasetEntry(name="hf-one", hf="org/a", split="train"),
        DatasetEntry(
            name="raw-one",
            url="https://raw.githubusercontent.com/o/r/m/a.jsonl",
        ),
        DatasetEntry(
            name="tree-one",
            url="https://github.com/o/r1",
            ext=(".jsonl",),
        ),
        DatasetEntry(
            name="clone-one",
            url="https://github.com/o/r2",
            mode="clone",
            lfs=True,
        ),
    ]
    manifest = _make_manifest(entries, tmp_path)
    result = runner.run_manifest(manifest, log=lambda _msg: None)

    assert sorted(result.succeeded) == ["clone-one", "hf-one", "raw-one", "tree-one"]
    assert result.failed == []
    kinds = [c[0] for c in calls]
    assert kinds == ["hf", "raw", "tree", "clone"]
    # Tokens were resolved from env.
    assert calls[0][1]["token"] == "HF_TOKEN_VALUE"
    assert calls[1][1]["token"] == "GH_TOKEN_VALUE"
    assert calls[2][1]["token"] == "GH_TOKEN_VALUE"
    assert calls[3][1]["token"] == "GH_TOKEN_VALUE"


def test_runner_resume_skips_existing(monkeypatch, tmp_path: Path) -> None:
    import convmerge.fetch.hf as hfmod

    hits: list[str] = []

    def fake_hf(*a, **kw):
        hits.append("called")
        raise AssertionError("should have been skipped")

    monkeypatch.setattr(hfmod, "download_hf_dataset", fake_hf)

    # Pre-populate the expected target so resume should skip it.
    target = tmp_path / "hf-one.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    runner._write_completion_marker(target)

    entries = [DatasetEntry(name="hf-one", hf="org/a")]
    manifest = _make_manifest(entries, tmp_path)
    result = runner.run_manifest(manifest, log=lambda _msg: None)
    assert result.skipped == ["hf-one"]
    assert hits == []


def test_runner_resume_refetches_without_completion_marker(monkeypatch, tmp_path: Path) -> None:
    import convmerge.fetch.hf as hfmod

    calls: list[str] = []
    target = tmp_path / "hf-one.jsonl"
    target.write_text("truncated\n", encoding="utf-8")

    def fake_hf(*_args, **_kwargs):
        calls.append("called")
        target.write_text('{"fresh": true}\n', encoding="utf-8")
        return target

    monkeypatch.setattr(hfmod, "download_hf_dataset", fake_hf)
    manifest = _make_manifest([DatasetEntry(name="hf-one", hf="org/a")], tmp_path)

    result = runner.run_manifest(manifest, log=lambda _msg: None)

    assert result.succeeded == ["hf-one"]
    assert result.skipped == []
    assert calls == ["called"]
    assert (tmp_path / "hf-one.jsonl.fetch.json").is_file()


def test_runner_resume_refetches_when_marked_output_changes(monkeypatch, tmp_path: Path) -> None:
    import convmerge.fetch.hf as hfmod

    calls: list[str] = []
    target = tmp_path / "hf-one.jsonl"
    target.write_text('{"complete": true}\n', encoding="utf-8")
    runner._write_completion_marker(target)
    target.write_text("truncated\n", encoding="utf-8")

    def fake_hf(*_args, **_kwargs):
        calls.append("called")
        target.write_text('{"repaired": true}\n', encoding="utf-8")
        return target

    monkeypatch.setattr(hfmod, "download_hf_dataset", fake_hf)
    manifest = _make_manifest([DatasetEntry(name="hf-one", hf="org/a")], tmp_path)

    result = runner.run_manifest(manifest, log=lambda _msg: None)

    assert result.succeeded == ["hf-one"]
    assert result.skipped == []
    assert calls == ["called"]


def test_runner_continues_on_error(monkeypatch, tmp_path: Path) -> None:
    import convmerge.fetch.hf as hfmod

    def boom(*a, **kw):
        raise RuntimeError("download failed")

    monkeypatch.setattr(hfmod, "download_hf_dataset", boom)

    entries = [
        DatasetEntry(name="ds1", hf="org/a"),
        DatasetEntry(name="ds2", hf="org/b"),
    ]
    manifest = _make_manifest(entries, tmp_path)
    result = runner.run_manifest(manifest, log=lambda _msg: None)
    assert [n for n, _ in result.failed] == ["ds1", "ds2"]
    assert result.succeeded == []


def test_runner_on_error_fail_raises(monkeypatch, tmp_path: Path) -> None:
    import convmerge.fetch.hf as hfmod

    def boom(*a, **kw):
        raise RuntimeError("nope")

    monkeypatch.setattr(hfmod, "download_hf_dataset", boom)

    manifest = Manifest(
        defaults=Defaults(output_root=str(tmp_path), on_error="fail", resume=True),
        datasets=(DatasetEntry(name="ds1", hf="org/a"),),
    )
    with pytest.raises(RuntimeError):
        runner.run_manifest(manifest, log=lambda _msg: None)


def test_runner_only_filter(monkeypatch, tmp_path: Path) -> None:
    import convmerge.fetch.hf as hfmod

    seen: list[str] = []

    def fake_hf(dataset_id, dst, *, config=None, split=None, token=None, max_rows=None):
        seen.append(dataset_id)
        Path(dst).write_text("{}\n", encoding="utf-8")
        return Path(dst)

    monkeypatch.setattr(hfmod, "download_hf_dataset", fake_hf)

    entries = [
        DatasetEntry(name="keep", hf="org/keep"),
        DatasetEntry(name="drop", hf="org/drop"),
    ]
    manifest = _make_manifest(entries, tmp_path)
    runner.run_manifest(manifest, only=["keep"], log=lambda _msg: None)
    assert seen == ["org/keep"]


def test_completion_snapshot_ignores_git_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "data.jsonl").write_text("{}\n", encoding="utf-8")
    (repo / ".git" / "FETCH_HEAD").write_text("a", encoding="utf-8")
    before = runner._completion_snapshot(repo)
    (repo / ".git" / "FETCH_HEAD").write_text("b", encoding="utf-8")
    assert runner._completion_snapshot(repo) == before
    assert [f["path"] for f in before["files"]] == ["data.jsonl"]


def test_runner_max_rows_passes_through_and_scopes_resume(monkeypatch, tmp_path: Path) -> None:
    import convmerge.fetch.hf as hfmod

    calls: list[int | None] = []

    def fake_hf(dataset_id, dst, *, config=None, split=None, token=None, max_rows=None):
        calls.append(max_rows)
        rows = max_rows or 5
        Path(dst).write_text("".join("{}\n" for _ in range(rows)), encoding="utf-8")
        return Path(dst)

    monkeypatch.setattr(hfmod, "download_hf_dataset", fake_hf)
    entries = [DatasetEntry(name="hf-one", hf="org/a", max_rows=3)]
    manifest = _make_manifest(entries, tmp_path)

    runner.run_manifest(manifest, log=lambda _m: None)
    marker = json.loads((tmp_path / "hf-one.jsonl.fetch.json").read_text())
    assert marker["max_rows"] == 3
    # Same sample size: resume skips.
    assert runner.run_manifest(manifest, log=lambda _m: None).skipped == ["hf-one"]
    # A full fetch (or another size) must not reuse the sample.
    runner.run_manifest(manifest, log=lambda _m: None, max_rows=10)
    full = _make_manifest([DatasetEntry(name="hf-one", hf="org/a")], tmp_path)
    runner.run_manifest(full, log=lambda _m: None)
    assert calls == [3, 10, None]
    assert "max_rows" not in json.loads((tmp_path / "hf-one.jsonl.fetch.json").read_text())


def test_runner_max_rows_rejects_json_array_raw(monkeypatch, tmp_path: Path) -> None:
    entries = [
        DatasetEntry(name="arr", url="https://raw.githubusercontent.com/o/r/m/a.json", max_rows=5)
    ]
    result = runner.run_manifest(_make_manifest(entries, tmp_path), log=lambda _m: None)
    assert result.failed and "line-delimited" in result.failed[0][1]


def test_manifest_max_rows_validation() -> None:
    from convmerge.fetch.manifest import _from_dict

    ok = _from_dict({"datasets": [{"name": "a", "hf": "o/d", "max_rows": 100}]})
    assert ok.datasets[0].max_rows == 100
    for bad in (0, -1, "10", True):
        with pytest.raises(ValueError, match="max_rows"):
            _from_dict({"datasets": [{"name": "a", "hf": "o/d", "max_rows": bad}]})
    with pytest.raises(ValueError, match="mode: clone"):
        _from_dict(
            {
                "datasets": [
                    {"name": "a", "url": "https://github.com/o/r", "mode": "clone", "max_rows": 5}
                ]
            }
        )


def test_hf_max_rows_streams_only_n_rows(monkeypatch, tmp_path: Path) -> None:
    import sys
    import types

    from convmerge.fetch.hf import download_hf_dataset

    seen: dict = {}

    class _Stream:
        def take(self, n):
            seen["take"] = n
            return iter([{"a": 1, "img": object.__new__(object)}, {"a": 2}][:n])

    def load_dataset(dataset_id, **kw):
        seen["kw"] = kw
        return _Stream()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    dst = tmp_path / "o.jsonl"
    download_hf_dataset("org/d", dst, split="train", max_rows=1, token="T")
    assert seen["kw"] == {"split": "train", "token": "T", "streaming": True}
    assert seen["take"] == 1
    row = json.loads(dst.read_text())
    assert row["a"] == 1 and isinstance(row["img"], str)
    assert [p.name for p in tmp_path.iterdir()] == ["o.jsonl"]
