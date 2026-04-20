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
