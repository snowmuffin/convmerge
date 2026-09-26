"""Thin wrapper over the system ``git`` (and optional ``git-lfs``) binaries."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from convmerge.fetch.auth import redact_url

# Hosts that receive the token. Anything else is cloned anonymously.
_TOKEN_HOSTS = ("github.com", "huggingface.co")


class GitNotFoundError(RuntimeError):
    """Raised when ``git`` is required but not installed on PATH."""


class GitLfsNotFoundError(RuntimeError):
    """Raised when ``git-lfs`` is requested via ``lfs: true`` but not installed."""


class GitCommandError(RuntimeError):
    """Raised when a git command exits non-zero (message is credential-free)."""


def clone_repo(
    url: str,
    dst_dir: str | Path,
    *,
    token: str | None = None,
    lfs: bool = False,
) -> Path:
    """Clone ``url`` into ``dst_dir`` (or ``git pull`` if it is already a repo).

    When ``token`` is provided and the URL targets ``github.com`` / HuggingFace,
    it is sent as an HTTP ``Authorization`` header scoped to that host. The
    header is passed through git's environment-based config
    (``GIT_CONFIG_COUNT``), so the token never appears in the command line,
    the clone URL, ``.git/config``, or error messages. For any other host the
    token is ignored.
    """
    if shutil.which("git") is None:
        raise GitNotFoundError(
            "git binary not found on PATH. Install git to use 'mode: clone' entries."
        )

    dst = Path(dst_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    clean_url = redact_url(url)
    env = _auth_env(clean_url, token)

    if (dst / ".git").is_dir():
        _scrub_stored_credentials(dst)
        _run(["git", "-C", str(dst), "pull", "--ff-only"], env=env)
    else:
        _run(["git", "clone", clean_url, str(dst)], env=env)

    if lfs:
        if shutil.which("git-lfs") is None:
            raise GitLfsNotFoundError(
                "git-lfs binary not found on PATH but entry requested lfs: true. "
                "Install git-lfs or drop the flag."
            )
        _run(["git", "-C", str(dst), "lfs", "pull"], env=env)

    return dst


def _token_host(url: str) -> str | None:
    """Return ``scheme://host/`` when ``url`` targets a token-eligible host."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in _TOKEN_HOSTS):
        return None
    return f"{parsed.scheme or 'https'}://{parsed.netloc}/"


def _auth_env(url: str, token: str | None) -> dict[str, str] | None:
    """Build a child-process environment carrying a host-scoped auth header."""
    if not token:
        return None
    prefix = _token_host(url)
    if prefix is None:
        return None
    # ``user`` is ignored by both hosts; the token acts as the password.
    basic = base64.b64encode(f"user:{token}".encode()).decode("ascii")
    env = dict(os.environ)
    idx = int(env.get("GIT_CONFIG_COUNT", "0") or 0)
    env[f"GIT_CONFIG_KEY_{idx}"] = f"http.{prefix}.extraHeader"
    env[f"GIT_CONFIG_VALUE_{idx}"] = f"Authorization: Basic {basic}"
    env["GIT_CONFIG_COUNT"] = str(idx + 1)
    # Never fall back to an interactive username/password prompt.
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _scrub_stored_credentials(dst: Path) -> None:
    """Remove a token that older convmerge versions embedded in the remote URL."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(dst), "remote", "get-url", "origin"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return
    current = (proc.stdout or "").strip()
    if proc.returncode != 0 or not current:
        return
    cleaned = redact_url(current)
    if cleaned != current:
        _run(["git", "-C", str(dst), "remote", "set-url", "origin", cleaned])


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        shown = " ".join(redact_url(part) for part in cmd)
        raise GitCommandError(f"git command failed (exit {e.returncode}): {shown}") from None
