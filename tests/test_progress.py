"""Tests for convmerge.progress (issue #18)."""

from __future__ import annotations

import io

from convmerge.progress import ProgressReporter, progress_enabled


def test_progress_enabled_flag_wins() -> None:
    assert progress_enabled(True) is True


def test_progress_enabled_env(monkeypatch) -> None:
    monkeypatch.setenv("CONVMERGE_PROGRESS", "1")
    assert progress_enabled(False) is True
    monkeypatch.setenv("CONVMERGE_PROGRESS", "yes")
    assert progress_enabled(None) is True


def test_progress_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CONVMERGE_PROGRESS", raising=False)
    assert progress_enabled(False) is False
    assert progress_enabled(None) is False


def test_progress_enabled_falsey_env(monkeypatch) -> None:
    monkeypatch.setenv("CONVMERGE_PROGRESS", "0")
    assert progress_enabled(False) is False


def test_reporter_disabled_emits_nothing() -> None:
    stream = io.StringIO()
    r = ProgressReporter("x", every=1, enabled=False, stream=stream)
    for _ in range(5):
        r.update()
    r.done()
    assert stream.getvalue() == ""


def test_reporter_emits_at_interval_and_done() -> None:
    stream = io.StringIO()
    r = ProgressReporter("dedupe a.jsonl", every=2, enabled=True, stream=stream)
    for _ in range(3):
        r.update()
    r.done()
    out = stream.getvalue()
    lines = out.strip().splitlines()
    # One interval line at count==2, plus a final "done" line at count==3.
    assert any(line.startswith("[progress]") and "2 rows" in line for line in lines)
    assert any(line.startswith("[done]") and "3 rows" in line for line in lines)


def test_reporter_done_not_duplicated_on_exact_interval() -> None:
    stream = io.StringIO()
    r = ProgressReporter("x", every=2, enabled=True, stream=stream)
    r.update()
    r.update()  # emits at count==2
    r.done()  # count == last_emitted, should not emit again
    assert stream.getvalue().count("rows") == 1
