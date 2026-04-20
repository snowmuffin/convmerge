"""GitHub fetchers: single raw URL + recursive Trees API with extension filter.

Uses stdlib ``urllib.request`` so core stays dependency-free; only PyYAML is
needed for manifest parsing (``[fetch]`` extra).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

_GITHUB_REPO_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/?#]+?)(?:\.git)?(?:/.*)?$"
)

_DEFAULT_TIMEOUT = 60


class GitHubFetchError(RuntimeError):
    """Raised when a GitHub API or download call fails."""


def download_raw_file(url: str, dst: str | Path, *, token: str | None = None) -> Path:
    """Download one raw URL (``raw.githubusercontent.com`` or similar) to ``dst``.

    Parent directories are created. Any ``Authorization`` header is only sent
    when ``token`` is provided.
    """
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"token {token}")
    req.add_header("User-Agent", "convmerge-fetch/0.2")

    try:
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        raise GitHubFetchError(f"HTTP {e.code} fetching {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise GitHubFetchError(f"Network error fetching {url}: {e.reason}") from e

    dst_p.write_bytes(data)
    return dst_p


def parse_repo_url(url: str) -> tuple[str, str]:
    """Return ``(owner, repo)`` for a ``github.com/owner/repo`` URL."""
    m = _GITHUB_REPO_RE.match(url.strip())
    if not m:
        raise ValueError(f"Not a GitHub repo URL: {url!r}")
    return m.group("owner"), m.group("repo")


def fetch_repo_tree_files(
    repo_url: str,
    dst_dir: str | Path,
    *,
    ext: tuple[str, ...] = (),
    token: str | None = None,
    branch: str | None = None,
) -> list[Path]:
    """Pull files from a GitHub repo via the Trees API (no full clone).

    Only files whose lowered path ends with one of ``ext`` are downloaded. When
    ``ext`` is empty, every blob in the tree is downloaded (use with care for
    large repos). Returns the list of downloaded file paths.
    """
    owner, repo = parse_repo_url(repo_url)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    if branch is None:
        branch = _get_default_branch(owner, repo, token=token)

    tree = _get_tree(owner, repo, branch, token=token)
    ext_lower = tuple(e.lower() for e in ext)

    out: list[Path] = []
    raw_base = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/"
    for node in tree.get("tree", []):
        if node.get("type") != "blob":
            continue
        path = node.get("path") or ""
        if ext_lower and not any(path.lower().endswith(e) for e in ext_lower):
            continue
        # Mirror repo hierarchy on disk but replace separators so the caller
        # gets a flat, file-manager-friendly directory when requested.
        local_name = path.replace("/", "_")
        dest = dst / local_name
        download_raw_file(raw_base + path, dest, token=token)
        out.append(dest)
    return out


def _get_default_branch(owner: str, repo: str, *, token: str | None) -> str:
    data = _github_api_json(f"https://api.github.com/repos/{owner}/{repo}", token=token)
    return data.get("default_branch") or "main"


def _get_tree(owner: str, repo: str, branch: str, *, token: str | None) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    return _github_api_json(url, token=token)


def _github_api_json(url: str, *, token: str | None) -> dict:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "convmerge-fetch/0.2")
    if token:
        req.add_header("Authorization", f"token {token}")
    try:
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise GitHubFetchError(f"HTTP {e.code} calling {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise GitHubFetchError(f"Network error calling {url}: {e.reason}") from e
