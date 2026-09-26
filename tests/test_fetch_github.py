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
        self._buf = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

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


def test_download_raw_file_rejects_lfs_pointer(monkeypatch, tmp_path: Path) -> None:
    pointer = b"version https://git-lfs.github.com/spec/v1\n"

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(pointer)

    monkeypatch.setattr(gh.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(gh.LfsPointerError, match="mode: clone with lfs: true"):
        gh.download_raw_file(
            "https://raw.githubusercontent.com/o/r/m/data.jsonl",
            tmp_path / "data.jsonl",
            resolve_lfs=False,
        )
    with pytest.raises(gh.LfsPointerError):
        gh.download_raw_file("https://example.com/data.jsonl", tmp_path / "data.jsonl")
    assert not (tmp_path / "data.jsonl").exists()
    assert list(tmp_path.iterdir()) == []


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


def _capture_headers(monkeypatch) -> list[dict[str, str]]:
    seen: list[dict[str, str]] = []

    def fake_urlopen(req, timeout=None):
        seen.append({k.lower(): v for k, v in req.header_items()})
        return _FakeResponse(b"{}")

    monkeypatch.setattr(gh.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_download_raw_file_no_token_for_non_github_host(monkeypatch, tmp_path: Path) -> None:
    seen = _capture_headers(monkeypatch)
    gh.download_raw_file("https://example.com/data.jsonl", tmp_path / "a.jsonl", token="abc")
    gh.download_raw_file(
        "https://raw.githubusercontent.com.evil.test/x.jsonl", tmp_path / "b.jsonl", token="abc"
    )
    assert all("authorization" not in h for h in seen)


def test_github_token_is_not_forwarded_on_redirect() -> None:
    import urllib.request

    req = urllib.request.Request("https://raw.githubusercontent.com/o/r/m/a.jsonl")
    gh._add_auth(req, "abc")
    assert req.get_header("Authorization") == "token abc"
    redirected = urllib.request.HTTPRedirectHandler().redirect_request(
        req, None, 302, "Found", {}, "https://other.example/blob"
    )
    assert redirected is not None
    assert "abc" not in str(dict(redirected.header_items()))


POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"ab" * 32 + b"\nsize 42\n"


def test_download_raw_file_resolves_lfs_via_batch_api(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str, dict, bytes | None]] = []

    def fake_urlopen(req, timeout=None):
        headers = {k.lower(): v for k, v in req.header_items()}
        calls.append((req.get_method(), req.full_url, headers, req.data))
        if req.full_url.startswith("https://raw.githubusercontent.com/"):
            return _FakeResponse(POINTER)
        if req.full_url.endswith("/info/lfs/objects/batch"):
            body = {
                "objects": [
                    {
                        "oid": "ab" * 32,
                        "size": 42,
                        "actions": {
                            "download": {
                                "href": "https://lfs.example/obj",
                                "header": {"X-Sig": "s1"},
                            }
                        },
                    }
                ]
            }
            return _FakeResponse(json.dumps(body).encode())
        if req.full_url == "https://lfs.example/obj":
            return _FakeResponse(b'{"a": 1}\n{"a": 2}\n{"a": 3}\n')
        raise AssertionError(req.full_url)

    monkeypatch.setattr(gh.urllib.request, "urlopen", fake_urlopen)
    dst = tmp_path / "d.jsonl"
    gh.download_raw_file(
        "https://raw.githubusercontent.com/org/repo/main/data/d.jsonl", dst, token="TOK", max_rows=2
    )
    assert dst.read_bytes() == b'{"a": 1}\n{"a": 2}\n'

    _, raw_url, raw_headers, _ = calls[0]
    assert raw_headers["authorization"] == "token TOK"
    method, batch_url, batch_headers, body = calls[1]
    assert (method, batch_url) == ("POST", "https://github.com/org/repo.git/info/lfs/objects/batch")
    assert json.loads(body)["objects"] == [{"oid": "ab" * 32, "size": 42}]
    assert batch_headers["authorization"].startswith("Basic ")
    _, obj_url, obj_headers, _ = calls[2]
    assert obj_headers.get("x-sig") == "s1"
    assert "authorization" not in obj_headers  # the token never reaches the object store


def test_lfs_batch_error_is_reported(monkeypatch, tmp_path: Path) -> None:
    def fake_urlopen(req, timeout=None):
        if req.full_url.endswith("/batch"):
            return _FakeResponse(
                json.dumps(
                    {"objects": [{"error": {"code": 404, "message": "Object does not exist"}}]}
                ).encode()
            )
        return _FakeResponse(POINTER)

    monkeypatch.setattr(gh.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(gh.GitHubFetchError, match="Object does not exist"):
        gh.download_raw_file(
            "https://raw.githubusercontent.com/o/r/m/x.jsonl", tmp_path / "x.jsonl"
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("max_rows", "expected"),
    [(None, b"1\n2\n3\n"), (1, b"1\n"), (3, b"1\n2\n3\n"), (10, b"1\n2\n3\n")],
)
def test_download_raw_file_max_rows(monkeypatch, tmp_path: Path, max_rows, expected) -> None:
    monkeypatch.setattr(gh, "_CHUNK", 2)  # force row boundaries across chunks
    monkeypatch.setattr(
        gh.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(b"1\n2\n3\n")
    )
    dst = tmp_path / "x.jsonl"
    gh.download_raw_file("https://raw.githubusercontent.com/o/r/m/x.jsonl", dst, max_rows=max_rows)
    assert dst.read_bytes() == expected
