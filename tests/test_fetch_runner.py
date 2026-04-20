"""End-to-end runner tests (all backends mocked)."""

from __future__ import annotations

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

    def fake_hf(dataset_id, dst, *, config=None, split=None, token=None):
        calls.append(("hf", {"id": dataset_id, "dst": str(dst), "token": token}))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("{}\n", encoding="utf-8")
        return Path(dst)

    def fake_raw(url, dst, *, token=None):
        calls.append(("raw", {"url": url, "dst": str(dst), "token": token}))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("{}\n", encoding="utf-8")
        return Path(dst)

    def fake_tree(url, dst, *, ext=(), token=None):
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

    entries = [DatasetEntry(name="hf-one", hf="org/a")]
    manifest = _make_manifest(entries, tmp_path)
    result = runner.run_manifest(manifest, log=lambda _msg: None)
    assert result.skipped == ["hf-one"]
    assert hits == []


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

    def fake_hf(dataset_id, dst, *, config=None, split=None, token=None):
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
