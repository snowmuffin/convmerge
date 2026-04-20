"""Tests for GitHub raw + Trees API fetching (urllib mocked, no network)."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from convmerge.fetch import github as gh


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_parse_repo_url() -> None:
    assert gh.parse_repo_url("https://github.com/Owner/Repo") == ("Owner", "Repo")
    assert gh.parse_repo_url("https://github.com/Owner/Repo.git") == ("Owner", "Repo")
    assert gh.parse_repo_url("https://github.com/Owner/Repo/tree/main") == ("Owner", "Repo")


def test_parse_repo_url_invalid() -> None:
    with pytest.raises(ValueError):
        gh.parse_repo_url("https://gitlab.com/o/r")


def test_download_raw_file_writes_bytes(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        return _FakeResponse(b"hello world")

    monkeypatch.setattr(gh.urllib.request, "urlopen", fake_urlopen)

    dst = tmp_path / "nested" / "out.jsonl"
    gh.download_raw_file("https://raw.githubusercontent.com/o/r/m/a.jsonl", dst, token="abc")
    assert dst.read_bytes() == b"hello world"
    assert captured["url"] == "https://raw.githubusercontent.com/o/r/m/a.jsonl"
    # Header names are title-cased by urllib.
    headers = {k.lower(): v for k, v in captured["headers"].items()}  # type: ignore[union-attr]
    assert headers["authorization"] == "token abc"


def test_download_raw_file_no_auth_when_no_token(monkeypatch, tmp_path: Path) -> None:
    def fake_urlopen(req, timeout=None):
        headers = {k.lower() for k, _ in req.header_items()}
        assert "authorization" not in headers
        return _FakeResponse(b"x")

    monkeypatch.setattr(gh.urllib.request, "urlopen", fake_urlopen)
    gh.download_raw_file("https://raw.githubusercontent.com/o/r/m/a.jsonl", tmp_path / "a")


def test_download_raw_file_http_error(monkeypatch, tmp_path: Path) -> None:
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))

    monkeypatch.setattr(gh.urllib.request, "urlopen", boom)
    with pytest.raises(gh.GitHubFetchError) as exc:
        gh.download_raw_file("https://raw.githubusercontent.com/o/r/m/a.jsonl", tmp_path / "a")
    assert "404" in str(exc.value)


def test_fetch_repo_tree_files_filters_by_ext(monkeypatch, tmp_path: Path) -> None:
    tree_response = {
        "tree": [
            {"type": "blob", "path": "data/a.jsonl"},
            {"type": "blob", "path": "data/b.txt"},
            {"type": "tree", "path": "data"},
            {"type": "blob", "path": "README.md"},
            {"type": "blob", "path": "sub/dir/c.jsonl"},
        ]
    }

    calls: list[str] = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        calls.append(url)
        if url.endswith("/repos/owner/repo"):
            return _FakeResponse(json.dumps({"default_branch": "main"}).encode())
        if "/git/trees/" in url:
            return _FakeResponse(json.dumps(tree_response).encode())
        return _FakeResponse(b"FILE:" + url.encode())

    monkeypatch.setattr(gh.urllib.request, "urlopen", fake_urlopen)

    downloaded = gh.fetch_repo_tree_files(
        "https://github.com/owner/repo",
        tmp_path / "out",
        ext=(".jsonl",),
        token="TOK",
    )
    names = sorted(p.name for p in downloaded)
    assert names == ["data_a.jsonl", "sub_dir_c.jsonl"]
    # Downloaded files exist and contain the mocked body.
    for p in downloaded:
        assert p.read_bytes().startswith(b"FILE:")
    # Tree API was queried with recursive=1.
    assert any("/git/trees/main?recursive=1" in url for url in calls)
