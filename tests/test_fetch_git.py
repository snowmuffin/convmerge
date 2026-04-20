"""Tests for the git / git-lfs wrapper (subprocess mocked)."""

from __future__ import annotations

from pathlib import Path

import pytest

from convmerge.fetch import git as gitmod


def test_clone_repo_invokes_git(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, check=True):
        calls.append(list(cmd))

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(gitmod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(gitmod.subprocess, "run", fake_run)

    gitmod.clone_repo("https://github.com/owner/repo", tmp_path / "dst")
    assert calls == [["git", "clone", "https://github.com/owner/repo", str(tmp_path / "dst")]]


def test_clone_repo_injects_token_for_github(monkeypatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []

    def fake_run(cmd, check=True):
        captured.append(list(cmd))

    monkeypatch.setattr(gitmod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(gitmod.subprocess, "run", fake_run)

    gitmod.clone_repo("https://github.com/owner/repo", tmp_path / "dst", token="TOK")
    url_arg = captured[0][2]
    assert "TOK@github.com" in url_arg
    assert url_arg.startswith("https://user:TOK@github.com/")


def test_clone_repo_does_not_inject_token_for_other_hosts(monkeypatch, tmp_path: Path) -> None:
    captured: list[list[str]] = []

    def fake_run(cmd, check=True):
        captured.append(list(cmd))

    monkeypatch.setattr(gitmod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(gitmod.subprocess, "run", fake_run)

    gitmod.clone_repo("https://gitlab.com/o/r", tmp_path / "dst", token="TOK")
    assert captured[0][2] == "https://gitlab.com/o/r"


def test_clone_repo_lfs_pull(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, check=True):
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
