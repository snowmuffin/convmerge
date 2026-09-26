"""Validation rules, convert --on-invalid / --report, and the validate command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.convert import ConvertStats, InvalidExampleError, convert_file, validate_file
from convmerge.models import ChatMessage, ContentPart, ToolCall, TrainingExample
from convmerge.validate import REASONS, validate_example


def _ex(*msgs: ChatMessage, issues: list[str] | None = None) -> TrainingExample:
    return TrainingExample(messages=list(msgs), issues=issues or [])


U = ChatMessage("user", "q")
A = ChatMessage("assistant", "a")


@pytest.mark.parametrize(
    ("example", "expected"),
    [
        (_ex(U, A), []),
        (_ex(), ["no_messages"]),
        (_ex(A), ["no_user"]),
        (_ex(U), ["no_assistant"]),
        (_ex(U, ChatMessage("assistant", "  ")), ["empty_message", "no_assistant"]),
        (_ex(U, ChatMessage("bot", "a")), ["unknown_role", "no_assistant"]),
        (_ex(U, ChatMessage("tool", "42"), A), ["orphan_tool_message"]),
        (
            _ex(
                U,
                ChatMessage("assistant", None, tool_calls=[ToolCall("f", "{}", id="c1")]),
                ChatMessage("tool", "42", tool_call_id="c2"),
                A,
            ),
            ["tool_call_id_mismatch"],
        ),
        # Tool calls without ids (LLaMA-Factory) pair by order.
        (
            _ex(
                U,
                ChatMessage("assistant", None, tool_calls=[ToolCall("f")]),
                ChatMessage("tool", "42"),
                A,
            ),
            [],
        ),
        # A tool call alone is a valid assistant target.
        (_ex(U, ChatMessage("assistant", None, tool_calls=[ToolCall("f")])), []),
        # Media-only user turn is not empty.
        (_ex(ChatMessage("user", [ContentPart("image", url="x.png")]), A), []),
        (_ex(U, A, issues=["unresolved_image"]), ["unresolved_image"]),
    ],
)
def test_validate_example(example: TrainingExample, expected: list[str]) -> None:
    assert validate_example(example) == expected


def test_every_reason_is_documented() -> None:
    assert set(REASONS) == {
        "no_messages",
        "unknown_role",
        "empty_message",
        "no_user",
        "no_assistant",
        "orphan_tool_message",
        "tool_call_id_mismatch",
    }


def _write(path: Path, *rows: dict) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


GOOD = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
BAD = {"messages": [{"role": "assistant", "content": "a"}]}


@pytest.mark.parametrize(
    ("mode", "written", "dropped", "kept"), [("drop", 1, 1, 0), ("keep", 2, 0, 1)]
)
def test_convert_on_invalid_drop_and_keep(tmp_path: Path, mode, written, dropped, kept) -> None:
    src = _write(tmp_path / "in.jsonl", GOOD, BAD)
    out = tmp_path / "out.jsonl"
    stats = ConvertStats()
    convert_file(
        src, out, adapter_name="chat", output_format="messages", stats=stats, on_invalid=mode
    )
    assert (stats.written, stats.dropped, stats.kept_invalid) == (written, dropped, kept)
    assert stats.drop_reasons == {"no_user": 1}
    assert stats.drop_lines == {"no_user": [2]}
    assert len(out.read_text().splitlines()) == written


def test_convert_on_invalid_fail(tmp_path: Path) -> None:
    src = _write(tmp_path / "in.jsonl", GOOD, BAD)
    with pytest.raises(InvalidExampleError, match="line 2.*no_user"):
        convert_file(
            src,
            tmp_path / "o.jsonl",
            adapter_name="chat",
            output_format="messages",
            on_invalid="fail",
        )
    with pytest.raises(ValueError, match="on_invalid"):
        convert_file(
            src,
            tmp_path / "o.jsonl",
            adapter_name="chat",
            output_format="messages",
            on_invalid="skip",
        )  # type: ignore[arg-type]


def test_drop_lines_keep_only_first_five(tmp_path: Path) -> None:
    src = _write(tmp_path / "in.jsonl", *[BAD] * 8)
    stats = ConvertStats()
    convert_file(
        src, tmp_path / "o.jsonl", adapter_name="chat", output_format="messages", stats=stats
    )
    assert stats.drop_reasons == {"no_user": 8}
    assert stats.drop_lines == {"no_user": [1, 2, 3, 4, 5]}


def test_validate_file_counts_without_writing(tmp_path: Path) -> None:
    src = _write(tmp_path / "in.jsonl", GOOD, BAD, GOOD)
    stats = validate_file(src)
    assert (stats.written, stats.dropped) == (2, 1)


def test_cli_convert_report_and_fail(tmp_path: Path, capsys) -> None:
    src = _write(tmp_path / "in.jsonl", GOOD, BAD)
    report = tmp_path / "r" / "report.json"
    main(
        [
            "convert",
            "-i",
            str(src),
            "-o",
            str(tmp_path / "o.jsonl"),
            "--from",
            "chat",
            "-f",
            "messages",
            "--report",
            str(report),
        ]
    )
    data = json.loads(report.read_text())
    assert data["dropped"] == 1
    assert data["reason_descriptions"] == {"no_user": REASONS["no_user"]}
    assert "dropped 1 examples (no_user=1)" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "convert",
                "-i",
                str(src),
                "-o",
                str(tmp_path / "o.jsonl"),
                "--from",
                "chat",
                "-f",
                "messages",
                "--on-invalid",
                "fail",
            ]
        )
    assert exc.value.code == 1
    assert "line 2" in capsys.readouterr().err


def test_cli_validate_exit_codes(tmp_path: Path, capsys) -> None:
    good = _write(tmp_path / "good.jsonl", GOOD)
    main(["validate", "-i", str(good)])
    assert json.loads(capsys.readouterr().out)["valid"] == 1

    bad = _write(tmp_path / "bad.jsonl", GOOD, BAD)
    with pytest.raises(SystemExit) as exc:
        main(["validate", "-i", str(bad)])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert (out["valid"], out["invalid"], out["drop_lines"]) == (1, 1, {"no_user": [2]})


def test_parallel_fail_mode_reports_line(tmp_path: Path, monkeypatch) -> None:
    import convmerge.convert as convmod

    monkeypatch.setattr(convmod, "_CHUNK_LINES", 1)
    src = _write(tmp_path / "in.jsonl", GOOD, GOOD, BAD, GOOD)
    with pytest.raises(InvalidExampleError, match="line 3.*no_user"):
        convert_file(
            src,
            tmp_path / "o.jsonl",
            adapter_name="chat",
            output_format="messages",
            on_invalid="fail",
            workers=2,
        )
