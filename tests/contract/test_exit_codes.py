"""CLI exit codes: 0 success, 1 the work failed, 2 invalid invocation (docs/stability.md)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import main


def _exit_code(argv: list[str]) -> int:
    try:
        main(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


@pytest.fixture
def files(tmp_path: Path) -> dict[str, str]:
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"},
                                 {"role": "assistant", "content": "a"}]}) + "\n"
    )  # fmt: skip
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"messages": [{"role": "user", "content": "q"}]}) + "\n")
    config = tmp_path / "broken.yaml"
    config.write_text("version: 1\nsources: [unclosed\n")
    return {
        "good": str(good),
        "bad": str(bad),
        "broken": str(config),
        "missing": str(tmp_path / "missing.jsonl"),
        "out": str(tmp_path / "out.jsonl"),
        "dir": str(tmp_path / "raw"),
    }


CASES = [
    # 0: success
    (0, ["validate", "-i", "{good}"]),
    (0, ["convert", "-i", "{good}", "-o", "{out}", "--from", "chat", "--format", "messages"]),
    # 1: the work failed
    (1, ["validate", "-i", "{bad}"]),
    (1, ["convert", "-i", "{missing}", "-o", "{out}", "--from", "chat", "--format", "messages"]),
    (1, ["convert", "-i", "{bad}", "-o", "{out}", "--from", "chat", "--format", "messages",
         "--on-invalid", "fail"]),
    (1, ["turns", "-i", "{missing}"]),
    (1, ["inspect", "-i", "{missing}"]),
    # 2: invalid invocation or configuration
    (2, ["convert", "-i", "{good}", "-o", "{out}", "--no-such-flag"]),
    (2, ["convert", "-i", "{good}", "-o", "{out}"]),
    (2, ["convert", "-i", "{good}", "-o", "{out}", "--preset", "{missing}"]),
    (2, ["turns", "-i", "{good}", "--single-out", "{out}"]),
    (2, ["fetch", "https://example.com/page", "-o", "{dir}"]),
    (2, ["fetch", "{missing}"]),
    (2, ["mix", "{missing}"]),
    (2, ["run", "{missing}"]),
    (2, ["convert", "-i", "{good}", "-o", "{out}", "--from", "chat", "--workers", "0"]),
]  # fmt: skip


@pytest.mark.parametrize(
    ("code", "argv"), CASES, ids=lambda v: " ".join(v)[:60] if isinstance(v, list) else str(v)
)
def test_exit_code(files: dict[str, str], code: int, argv: list[str], capsys) -> None:
    assert _exit_code([a.format(**files) for a in argv]) == code


@pytest.mark.parametrize("command", ["fetch", "mix", "run"])
def test_invalid_yaml_config_exits_2(files: dict[str, str], command: str) -> None:
    pytest.importorskip("yaml")
    assert _exit_code([command, files["broken"]]) == 2
