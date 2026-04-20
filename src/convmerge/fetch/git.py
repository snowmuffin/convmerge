"""Thin wrapper over the system ``git`` (and optional ``git-lfs``) binaries."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse


class GitNotFoundError(RuntimeError):
    """Raised when ``git`` is required but not installed on PATH."""


class GitLfsNotFoundError(RuntimeError):
    """Raised when ``git-lfs`` is requested via ``lfs: true`` but not installed."""


def clone_repo(
    url: str,
    dst_dir: str | Path,
    *,
    token: str | None = None,
    lfs: bool = False,
) -> Path:
    """Clone ``url`` into ``dst_dir`` (or ``git pull`` if it is already a repo).

    When ``token`` is provided and the URL targets ``github.com`` / HuggingFace,
    it is injected as the HTTPS basic-auth password for the clone call and
    **never** logged. For any other host the token is ignored.
    """
    if shutil.which("git") is None:
        raise GitNotFoundError(
            "git binary not found on PATH. Install git to use 'mode: clone' entries."
        )

    dst = Path(dst_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if (dst / ".git").is_dir():
        _run(["git", "-C", str(dst), "pull", "--ff-only"])
    else:
        clone_url = _maybe_inject_token(url, token)
        _run(["git", "clone", clone_url, str(dst)])

    if lfs:
        if shutil.which("git-lfs") is None:
            raise GitLfsNotFoundError(
                "git-lfs binary not found on PATH but entry requested lfs: true. "
                "Install git-lfs or drop the flag."
            )
        _run(["git", "-C", str(dst), "lfs", "pull"])

    return dst


def _maybe_inject_token(url: str, token: str | None) -> str:
    if not token:
        return url
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "github.com" not in host and "huggingface.co" not in host:
        return url
    scheme = parsed.scheme or "https"
    # ``user:`` is ignored by both hosts; the token acts as the password.
    auth_netloc = f"user:{token}@{parsed.netloc}"
    return f"{scheme}://{auth_netloc}{parsed.path}"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)
