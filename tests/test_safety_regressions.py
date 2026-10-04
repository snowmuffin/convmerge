"""Safety regressions fixed in 1.6.3."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from convmerge import ConvertStats, convert_file
from convmerge.cli import main
from convmerge.convert import InvalidExampleError
from convmerge.io import ReadStats, SamePathError, iter_jsonl
from convmerge.mix import MixSource, mix_files
from convmerge.split import split_jsonl


def _chat(user: str = "q", assistant: str = "a") -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("existing", [False, True])
def test_failed_convert_does_not_leave_partial_output(tmp_path: Path, existing: bool) -> None:
    src = _write_rows(
        tmp_path / "in.jsonl",
        [_chat("good", "answer"), {"messages": [{"role": "user", "content": "bad"}]}],
    )
    out = tmp_path / "out.jsonl"
    if existing:
        out.write_text("previous-success\n", encoding="utf-8")

    with pytest.raises(InvalidExampleError):
        convert_file(
            src,
            out,
            adapter_name="auto",
            output_format="messages",
            on_invalid="fail",
        )

    if existing:
        assert out.read_text(encoding="utf-8") == "previous-success\n"
    else:
        assert not out.exists()


def test_exact_split_refuses_duplicate_boundary(tmp_path: Path) -> None:
    src = _write_rows(tmp_path / "in.jsonl", [_chat(), _chat()])
    with pytest.raises(ValueError, match="would split rows with the same content/hash"):
        split_jsonl(src, tmp_path / "train.jsonl", tmp_path / "val.jsonl", val_rows=1)
    assert not (tmp_path / "train.jsonl").exists()
    assert not (tmp_path / "val.jsonl").exists()


def test_exact_split_refuses_key_group_boundary(tmp_path: Path) -> None:
    rows = [{"prompt": "same", "answer": "a"}, {"prompt": "same", "answer": "b"}]
    src = _write_rows(tmp_path / "in.jsonl", rows)
    with pytest.raises(ValueError, match="would split rows with the same content/hash"):
        split_jsonl(
            src,
            tmp_path / "train.jsonl",
            tmp_path / "val.jsonl",
            val_rows=1,
            keys=["prompt"],
        )


@pytest.mark.parametrize("workers", [1, 2])
def test_huge_json_integer_is_an_unreadable_line(tmp_path: Path, workers: int) -> None:
    get_limit = getattr(sys, "get_int_max_str_digits", None)
    if get_limit is None or get_limit() == 0:
        pytest.skip("this Python has no integer-string conversion limit")
    digits = max(5_000, get_limit() + 1)
    src = tmp_path / "in.jsonl"
    src.write_text(
        json.dumps(_chat("first", "ok"))
        + "\n"
        + '{"n": '
        + "1" * digits
        + "}\n"
        + json.dumps(_chat("last", "ok"))
        + "\n",
        encoding="utf-8",
    )

    stats = ConvertStats()
    convert_file(
        src,
        tmp_path / "out.jsonl",
        adapter_name="auto",
        output_format="messages",
        workers=workers,
        stats=stats,
    )
    assert (stats.written, stats.invalid_json, stats.first_invalid_line) == (2, 1, 2)

    read = ReadStats()
    assert len(list(iter_jsonl(src, stats=read))) == 2
    assert (read.invalid_json, read.first_invalid_line) == (1, 2)


def _oasst(mid: str, parent: str | None, role: str, text: str) -> dict:
    return {
        "message_tree_id": "tree",
        "message_id": mid,
        "parent_id": parent,
        "role": role,
        "text": text,
        "rank": 0,
        "deleted": False,
        "review_result": True,
    }


def test_mixed_openassistant_rows_match_with_workers(tmp_path: Path) -> None:
    rows = [
        _chat("ordinary", "row"),
        _oasst("root", None, "prompter", "tree question"),
        _oasst("answer", "root", "assistant", "tree answer"),
    ]
    src = _write_rows(tmp_path / "in.jsonl", rows)
    outputs = []
    stats_list = []
    for workers in (1, 2):
        out = tmp_path / f"out-{workers}.jsonl"
        stats = ConvertStats()
        convert_file(
            src,
            out,
            adapter_name="auto",
            output_format="messages",
            workers=workers,
            stats=stats,
        )
        outputs.append(out.read_bytes())
        stats_list.append(asdict(stats))

    assert outputs[0] == outputs[1]
    assert stats_list[0] == stats_list[1]
    assert stats_list[0]["written"] == 2
    assert stats_list[0]["grouped"] == 1


@pytest.mark.parametrize("through_symlink", [False, True])
def test_mix_sidecar_cannot_replace_a_source(tmp_path: Path, through_symlink: bool, capsys) -> None:
    output = tmp_path / "train.jsonl"
    source = tmp_path / "source.jsonl"
    _write_rows(source, [_chat(), _chat("q2", "a2")])
    sidecar = output.with_suffix(".mix.json")
    if through_symlink:
        sidecar.symlink_to(source)
        input_path = source
    else:
        source.rename(sidecar)
        input_path = sidecar
        source = sidecar
    before = source.read_bytes()

    with pytest.raises(SystemExit) as exc:
        main(["mix", "-i", f"{input_path}:1", "-o", str(output)])
    assert exc.value.code == 2
    assert source.read_bytes() == before
    assert "input file" in capsys.readouterr().err


def test_mix_config_cannot_be_its_output(tmp_path: Path, capsys) -> None:
    source = _write_rows(tmp_path / "source.jsonl", [_chat()])
    config = tmp_path / "mix.json"
    config.write_text(
        json.dumps(
            {
                "sources": [{"path": str(source), "weight": 1}],
                "output": str(config),
            }
        ),
        encoding="utf-8",
    )
    before = config.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["mix", str(config), "--no-recipe"])
    assert exc.value.code == 2
    assert config.read_bytes() == before
    assert "input file" in capsys.readouterr().err


def test_mix_library_refuses_source_output_and_nonfinite_weight(tmp_path: Path) -> None:
    source = _write_rows(tmp_path / "source.jsonl", [_chat()])
    with pytest.raises(SamePathError):
        mix_files([MixSource(source, 1)], source)
    with pytest.raises(ValueError, match="finite"):
        mix_files([MixSource(source, math.inf)], tmp_path / "out.jsonl")


def test_mix_cli_rejects_infinite_weight(tmp_path: Path, capsys) -> None:
    source = _write_rows(tmp_path / "source.jsonl", [_chat()])
    with pytest.raises(SystemExit) as exc:
        main(["mix", "-i", f"{source}:inf", "-o", str(tmp_path / "out.jsonl")])
    assert exc.value.code == 2
    assert "positive number" in capsys.readouterr().err


def test_convert_preset_cannot_be_its_output(tmp_path: Path, capsys) -> None:
    pytest.importorskip("yaml")
    source = _write_rows(tmp_path / "source.jsonl", [_chat()])
    preset = tmp_path / "preset.yaml"
    preset.write_text("adapter: auto\noutput_format: messages\n", encoding="utf-8")
    before = preset.read_bytes()

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "convert",
                "-i",
                str(source),
                "-o",
                str(preset),
                "--preset",
                str(preset),
            ]
        )
    assert exc.value.code == 2
    assert preset.read_bytes() == before
    assert "input file" in capsys.readouterr().err


def test_preset_validate_missing_file_is_invocation_error(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["preset", "validate", str(tmp_path / "missing.yaml")])
    assert exc.value.code == 2
    assert "preset file not found" in capsys.readouterr().err


def test_split_cli_explains_exact_group_conflict(tmp_path: Path, capsys) -> None:
    source = _write_rows(tmp_path / "source.jsonl", [_chat(), _chat()])
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "split",
                "-i",
                str(source),
                "-o",
                str(tmp_path / "train.jsonl"),
                "--val-rows",
                "1",
            ]
        )
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "would split rows with the same content/hash" in err
    assert "Traceback" not in err
