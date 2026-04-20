"""Tests for token resolution and URL redaction."""

from __future__ import annotations

from pathlib import Path

from convmerge.fetch.auth import TokenSpec, redact_url, resolve_token


def test_resolve_token_explicit_wins(monkeypatch, tmp_path: Path) -> None:
    file_p = tmp_path / "tok"
    file_p.write_text("FROM_FILE", encoding="utf-8")
    monkeypatch.setenv("CM_TOK", "FROM_ENV")
    spec = TokenSpec(env="CM_TOK", file=str(file_p))
    assert resolve_token(spec, explicit="FROM_CLI") == "FROM_CLI"


def test_resolve_token_file_over_env(monkeypatch, tmp_path: Path) -> None:
    file_p = tmp_path / "tok"
    file_p.write_text("FROM_FILE", encoding="utf-8")
    monkeypatch.setenv("CM_TOK", "FROM_ENV")
    spec = TokenSpec(env="CM_TOK", file=str(file_p))
    assert resolve_token(spec) == "FROM_FILE"


def test_resolve_token_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CM_TOK", "E")
    assert resolve_token(TokenSpec(env="CM_TOK")) == "E"


def test_resolve_token_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("CM_TOK", raising=False)
    assert resolve_token(TokenSpec(env="CM_TOK")) is None


def test_redact_url_strips_userinfo() -> None:
    assert redact_url("https://user:s3cr3t@github.com/org/repo") == "https://github.com/org/repo"
    assert redact_url("https://github.com/org/repo") == "https://github.com/org/repo"
    assert redact_url("http://tok@example.com") == "http://example.com"
