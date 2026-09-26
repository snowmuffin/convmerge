"""Tests for the git / git-lfs wrapper (subprocess mocked)."""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

import pytest

from convmerge.fetch import git as gitmod


def _record_runs(monkeypatch) -> list[tuple[list[str], dict | None]]:
    calls: list[tuple[list[str], dict | None]] = []

    def fake_run(cmd, check=True, env=None, **kw):
        calls.append((list(cmd), env))

        class _R:
            returncode = 0
            stdout = ""

        return _R()

    monkeypatch.setattr(gitmod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(gitmod.subprocess, "run", fake_run)
    return calls


def test_clone_repo_invokes_git(monkeypatch, tmp_path: Path) -> None:
    calls = _record_runs(monkeypatch)
    gitmod.clone_repo("https://github.com/owner/repo", tmp_path / "dst")
    assert calls == [
        (["git", "clone", "https://github.com/owner/repo", str(tmp_path / "dst")], None)
    ]


def test_clone_repo_token_never_on_command_line(monkeypatch, tmp_path: Path) -> None:
    calls = _record_runs(monkeypatch)
    gitmod.clone_repo("https://github.com/owner/repo", tmp_path / "dst", token="TOK")
    cmd, env = calls[0]
    assert cmd == ["git", "clone", "https://github.com/owner/repo", str(tmp_path / "dst")]
    assert all("TOK" not in part for part in cmd)
    assert env is not None
    idx = int(env["GIT_CONFIG_COUNT"]) - 1
    assert env[f"GIT_CONFIG_KEY_{idx}"] == "http.https://github.com/.extraHeader"
    assert env[f"GIT_CONFIG_VALUE_{idx}"] == "Authorization: Basic " + base64.b64encode(
        b"user:TOK"
    ).decode("ascii")
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_clone_repo_strips_userinfo_from_url(monkeypatch, tmp_path: Path) -> None:
    calls = _record_runs(monkeypatch)
    gitmod.clone_repo("https://user:OLD@github.com/o/r", tmp_path / "dst")
    assert calls[0][0][2] == "https://github.com/o/r"


def test_clone_repo_appends_to_existing_git_config_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "false")
    calls = _record_runs(monkeypatch)
    gitmod.clone_repo("https://huggingface.co/datasets/o/r", tmp_path / "dst", token="TOK")
    env = calls[0][1]
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert env["GIT_CONFIG_KEY_0"] == "core.autocrlf"
    assert env["GIT_CONFIG_KEY_1"] == "http.https://huggingface.co/.extraHeader"


def test_clone_repo_does_not_send_token_to_other_hosts(monkeypatch, tmp_path: Path) -> None:
    calls = _record_runs(monkeypatch)
    for url in ("https://gitlab.com/o/r", "https://notgithub.com/o/r"):
        gitmod.clone_repo(url, tmp_path / url.rsplit("/", 2)[-2], token="TOK")
    assert [env for _, env in calls] == [None, None]


def test_clone_repo_pull_uses_token_and_scrubs_stored_url(monkeypatch, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    (dst / ".git").mkdir(parents=True)
    calls: list[tuple[list[str], dict | None]] = []

    def fake_run(cmd, check=True, env=None, **kw):
        calls.append((list(cmd), env))

        class _R:
            returncode = 0
            stdout = "https://user:OLDTOK@github.com/o/r\n" if "get-url" in cmd else ""

        return _R()

    monkeypatch.setattr(gitmod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(gitmod.subprocess, "run", fake_run)
    gitmod.clone_repo("https://github.com/o/r", dst, token="TOK")
    cmds = [c for c, _ in calls]
    assert ["git", "-C", str(dst), "remote", "set-url", "origin", "https://github.com/o/r"] in cmds
    pull_env = next(env for c, env in calls if "pull" in c)
    assert pull_env is not None and pull_env["GIT_TERMINAL_PROMPT"] == "0"


def test_clone_repo_failure_message_has_no_token(monkeypatch, tmp_path: Path) -> None:
    def failing_run(cmd, check=True, env=None, **kw):
        raise subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(gitmod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(gitmod.subprocess, "run", failing_run)
    with pytest.raises(gitmod.GitCommandError) as exc:
        gitmod.clone_repo("https://user:TOK@github.com/o/r", tmp_path / "d", token="TOK")
    assert "TOK" not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_scrub_stored_credentials_real_git(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", "https://user:TOK@github.com/o/r"],
        check=True,
    )
    gitmod._scrub_stored_credentials(repo)
    url = subprocess.run(
        ["git", "-C", str(repo), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert url == "https://github.com/o/r"


def test_clone_repo_lfs_pull(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, check=True, env=None, **kw):
        calls.append(list(cmd))

    monkeypatch.setattr(gitmod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(gitmod.subprocess, "run", fake_run)

    dst = tmp_path / "dst"
    gitmod.clone_repo("https://github.com/o/r", dst, lfs=True)
    assert any(cmd[:4] == ["git", "-C", str(dst), "lfs"] for cmd in calls)


def test_clone_repo_missing_git(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gitmod.shutil, "which", lambda name: None)
    with pytest.raises(gitmod.GitNotFoundError):
        gitmod.clone_repo("https://github.com/o/r", tmp_path / "d")


def test_clone_repo_missing_lfs(monkeypatch, tmp_path: Path) -> None:
    def which(name: str) -> str | None:
        return "/usr/bin/git" if name == "git" else None

    monkeypatch.setattr(gitmod.shutil, "which", which)
    monkeypatch.setattr(gitmod.subprocess, "run", lambda *a, **kw: None)
    with pytest.raises(gitmod.GitLfsNotFoundError):
        gitmod.clone_repo("https://github.com/o/r", tmp_path / "d", lfs=True)
