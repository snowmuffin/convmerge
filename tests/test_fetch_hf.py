"""Tests for the HuggingFace fetch wrapper (no real downloads)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _install_fake_datasets(monkeypatch, calls: list[dict]) -> None:
    class _FakeDS:
        def to_json(self, path: str) -> None:
            calls.append({"event": "to_json", "path": path})
            Path(path).write_text("", encoding="utf-8")

    def fake_load_dataset(*args, **kwargs):
        calls.append({"event": "load_dataset", "args": args, "kwargs": kwargs})
        return _FakeDS()

    mod = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_download_hf_dataset_basic(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_datasets(monkeypatch, calls)

    from convmerge.fetch.hf import download_hf_dataset

    dst = tmp_path / "out.jsonl"
    out = download_hf_dataset("org/ds", dst, split="train")
    assert out == dst
    assert dst.is_file()
    assert calls[0]["args"] == ("org/ds",)
    assert calls[0]["kwargs"] == {"split": "train"}
    assert calls[1]["event"] == "to_json"


def test_download_hf_dataset_passes_config_and_token(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_datasets(monkeypatch, calls)

    from convmerge.fetch.hf import download_hf_dataset

    download_hf_dataset(
        "org/ds", tmp_path / "out.jsonl", config="sub", split="validation", token="T"
    )
    kwargs = calls[0]["kwargs"]
    assert kwargs["name"] == "sub"
    assert kwargs["split"] == "validation"
    assert kwargs["token"] == "T"


def test_download_hf_dataset_missing_datasets(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "datasets", None)
    from convmerge.fetch.hf import download_hf_dataset

    with pytest.raises(ImportError):
        download_hf_dataset("org/ds", tmp_path / "out.jsonl")


def test_download_hf_dataset_pins_revision(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_datasets(monkeypatch, calls)

    from convmerge.fetch.hf import download_hf_dataset

    download_hf_dataset("org/ds", tmp_path / "out.jsonl", revision="abc123")
    assert calls[0]["kwargs"]["revision"] == "abc123"


def test_manifest_revision_reaches_the_download_and_scopes_resume(
    monkeypatch, tmp_path: Path
) -> None:
    from convmerge.fetch import hf
    from convmerge.fetch.manifest import _from_dict as parse_manifest
    from convmerge.fetch.runner import run_manifest

    seen: list[str | None] = []

    def fake(dataset_id, dst, **kw):
        seen.append(kw.get("revision"))
        Path(dst).write_text('{"a": 1}\n', encoding="utf-8")
        return Path(dst)

    monkeypatch.setattr(hf, "download_hf_dataset", fake)
    entry = {"name": "one", "hf": "org/ds", "revision": "v1"}
    run_manifest(parse_manifest({"datasets": [entry]}), output_root=tmp_path)
    run_manifest(parse_manifest({"datasets": [entry]}), output_root=tmp_path)
    run_manifest(parse_manifest({"datasets": [{**entry, "revision": "v2"}]}), output_root=tmp_path)
    assert seen == ["v1", "v2"]


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        ({"name": "x", "url": "https://example.com/a.jsonl", "revision": "v1"}, "only applies"),
        ({"name": "x", "hf": "org/ds", "revision": ""}, "non-empty string"),
    ],
)
def test_manifest_revision_errors(entry: dict, error: str) -> None:
    from convmerge.fetch.manifest import _from_dict as parse_manifest

    with pytest.raises(ValueError, match=error):
        parse_manifest({"datasets": [entry]})
