"""GitHub fetchers: single raw URL + recursive Trees API with extension filter.

Uses stdlib ``urllib.request`` so core stays dependency-free; only PyYAML is
needed for manifest parsing (``[fetch]`` extra).
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from convmerge.fetch.auth import redact_url
from convmerge.lfs import LfsPointerError, is_lfs_pointer, parse_lfs_pointer

_GITHUB_REPO_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/?#]+?)(?:\.git)?(?:/.*)?$"
)

_DEFAULT_TIMEOUT = 60
_CHUNK = 1 << 20

# Hosts that may receive a GitHub token. Other URLs are fetched anonymously.
_GITHUB_TOKEN_HOSTS = frozenset({"github.com", "api.github.com", "raw.githubusercontent.com"})


class GitHubFetchError(RuntimeError):
    """Raised when a GitHub API or download call fails."""


def download_raw_file(
    url: str,
    dst: str | Path,
    *,
    token: str | None = None,
    max_rows: int | None = None,
    resolve_lfs: bool = True,
) -> Path:
    """Download one raw URL (``raw.githubusercontent.com`` or similar) to ``dst``.

    The body is streamed to disk. Parent directories are created. An
    ``Authorization`` header is only sent when ``token`` is provided and the
    URL is on a GitHub host; it is never forwarded across redirects.

    ``max_rows`` keeps only the first N lines (for line-delimited files such as
    JSONL) and stops downloading there. When a ``raw.githubusercontent.com``
    URL returns a Git LFS pointer, the real object is fetched through the Git
    LFS batch API (``resolve_lfs=True``, default) — no clone needed; other
    hosts, or ``resolve_lfs=False``, raise :class:`LfsPointerError`.
    """
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_p.with_name(f".{dst_p.name}.part")
    try:
        with _open(url, token=token) as resp:
            head = resp.read(1024)
            if is_lfs_pointer(head):
                pointer = head + resp.read(1024)
                lfs_url, headers = _resolve_lfs(url, pointer, token=token, resolve=resolve_lfs)
                with _open(lfs_url, headers=headers) as lfs_resp:
                    _stream(lfs_resp, b"", tmp, max_rows)
            else:
                _stream(resp, head, tmp, max_rows)
        tmp.replace(dst_p)
    finally:
        tmp.unlink(missing_ok=True)
    return dst_p


def _open(url: str, *, token: str | None = None, headers: dict[str, str] | None = None):
    req = urllib.request.Request(url)
    _add_auth(req, token)
    req.add_header("User-Agent", "convmerge-fetch/0.7")
    for k, v in (headers or {}).items():
        req.add_unredirected_header(k, v)
    try:
        return urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT)
    except urllib.error.HTTPError as e:
        raise GitHubFetchError(f"HTTP {e.code} fetching {redact_url(url)}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise GitHubFetchError(f"Network error fetching {redact_url(url)}: {e.reason}") from e


def _stream(resp, head: bytes, dst: Path, max_rows: int | None) -> None:
    """Write ``head`` + the rest of ``resp`` to ``dst``, stopping after ``max_rows`` lines."""
    lines = 0
    with dst.open("wb") as out:
        chunk = head or resp.read(_CHUNK)
        while chunk:
            if max_rows is not None:
                need = max_rows - lines
                count = chunk.count(b"\n")
                if count >= need:
                    cut = -1
                    for _ in range(need):
                        cut = chunk.index(b"\n", cut + 1)
                    out.write(chunk[: cut + 1])
                    return
                lines += count
            out.write(chunk)
            chunk = resp.read(_CHUNK)


_RAW_URL_RE = re.compile(
    r"^https?://raw\.githubusercontent\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/", re.IGNORECASE
)


def _resolve_lfs(
    url: str, pointer: bytes, *, token: str | None, resolve: bool
) -> tuple[str, dict[str, str]]:
    """Ask the repository's Git LFS server where the object behind ``pointer`` lives."""
    m = _RAW_URL_RE.match(url)
    if not resolve or m is None:
        raise LfsPointerError(
            f"{redact_url(url)} returned a Git LFS pointer instead of the dataset blob; "
            "fetch it from raw.githubusercontent.com (resolved automatically) or use "
            "mode: clone with lfs: true"
        )
    oid, size = parse_lfs_pointer(pointer)
    batch_url = f"https://github.com/{m['owner']}/{m['repo']}.git/info/lfs/objects/batch"
    body = json.dumps(
        {"operation": "download", "transfers": ["basic"], "objects": [{"oid": oid, "size": size}]}
    ).encode()
    req = urllib.request.Request(batch_url, data=body, method="POST")
    req.add_header("Accept", "application/vnd.git-lfs+json")
    req.add_header("Content-Type", "application/vnd.git-lfs+json")
    req.add_header("User-Agent", "convmerge-fetch/0.7")
    if token:
        basic = base64.b64encode(f"user:{token}".encode()).decode("ascii")
        req.add_unredirected_header("Authorization", f"Basic {basic}")
    try:
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise GitHubFetchError(f"HTTP {e.code} from Git LFS batch API for {batch_url}") from e
    except urllib.error.URLError as e:
        raise GitHubFetchError(f"Network error calling {batch_url}: {e.reason}") from e
    objects = data.get("objects") or [{}]
    obj = objects[0]
    if obj.get("error"):
        raise GitHubFetchError(f"Git LFS object {oid[:12]}: {obj['error'].get('message')}")
    action = (obj.get("actions") or {}).get("download") or {}
    href = action.get("href")
    if not href:
        raise GitHubFetchError(f"Git LFS batch API returned no download URL for {oid[:12]}")
    return href, {str(k): str(v) for k, v in (action.get("header") or {}).items()}


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
    max_rows: int | None = None,
) -> list[Path]:
    """Pull files from a GitHub repo via the Trees API (no full clone).

    Only files whose lowered path ends with one of ``ext`` are downloaded. When
    ``ext`` is empty, every blob in the tree is downloaded (use with care for
    large repos). ``max_rows`` samples the first N lines of each non-``.json``
    file; LFS-backed files are resolved like :func:`download_raw_file`.
    Returns the list of downloaded file paths.
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
        # .json files cannot be cut by lines; only line formats are sampled.
        rows = None if path.lower().endswith(".json") else max_rows
        download_raw_file(raw_base + path, dest, token=token, max_rows=rows)
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
    _add_auth(req, token)
    try:
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise GitHubFetchError(f"HTTP {e.code} calling {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise GitHubFetchError(f"Network error calling {url}: {e.reason}") from e


def _add_auth(req: urllib.request.Request, token: str | None) -> None:
    """Attach the token only for GitHub hosts, and never across redirects.

    urllib copies ordinary headers onto redirected requests, even to another
    host; "unredirected" headers are sent on the original request only.
    """
    if not token:
        return
    host = (urlparse(req.full_url).hostname or "").lower()
    if host not in _GITHUB_TOKEN_HOSTS:
        return
    req.add_unredirected_header("Authorization", f"token {token}")
