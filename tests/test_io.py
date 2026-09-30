"""Shared JSONL reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.io import JsonlDecodeError, ReadStats, iter_jsonl


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


def test_iter_jsonl_skips_blank_and_invalid_with_counts(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.jsonl", '﻿{"a": 1}\n\n  {bad\n[1]\r\n"s"  \n')
    stats = ReadStats()
    lines = list(iter_jsonl(p, stats=stats))
    assert [(x.number, x.raw, x.value) for x in lines] == [
        (1, '{"a": 1}', {"a": 1}),
        (4, "[1]", [1]),
        (5, '"s"', "s"),
    ]
    assert stats == ReadStats(lines_read=5, blank=1, invalid_json=1, first_invalid_line=3)


def test_iter_jsonl_on_invalid_callback(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.jsonl", '{"a": 1}\n{bad\n')
    seen: list[JsonlDecodeError] = []
    assert len(list(iter_jsonl(p, on_invalid=seen.append))) == 1
    assert [e.line_number for e in seen] == [2]
    assert "line 2" in str(seen[0])


def test_iter_jsonl_raise(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.jsonl", '{"a": 1}\n{bad\n')
    it = iter_jsonl(p, on_error="raise")
    assert next(it).value == {"a": 1}
    with pytest.raises(JsonlDecodeError, match="line 2"):
        next(it)


def test_iter_jsonl_rejects_unknown_policy(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.jsonl", "{}\n")
    with pytest.raises(ValueError, match="on_error"):
        list(iter_jsonl(p, on_error="ignore"))  # type: ignore[arg-type]


def test_convert_accepts_bom_prefixed_file(tmp_path: Path) -> None:
    from convmerge.convert import ConvertStats, convert_file

    src = _write(tmp_path / "in.jsonl", '﻿{"instruction": "a", "output": "b"}\n')
    stats = ConvertStats()
    convert_file(
        src, tmp_path / "o.jsonl", adapter_name="alpaca", output_format="messages", stats=stats
    )
    assert (stats.written, stats.invalid_json) == (1, 0)


def test_turns_reports_invalid_line_number(tmp_path: Path) -> None:
    from convmerge.normalize.turns import analyze_turn_distribution

    p = _write(tmp_path / "a.jsonl", '{"messages": []}\n{bad\n')
    with pytest.raises(JsonlDecodeError, match="line 2"):
        analyze_turn_distribution(p)


def test_fetch_runner_logs_to_stderr_by_default(tmp_path: Path, capsys) -> None:
    from convmerge.fetch.manifest import Defaults, Manifest
    from convmerge.fetch.runner import run_manifest

    run_manifest(Manifest(defaults=Defaults(output_root=str(tmp_path))))
    out = capsys.readouterr()
    assert out.out == ""
    assert "[done]" in out.err


def test_cli_prints_library_warnings_to_stderr(tmp_path: Path, capsys, monkeypatch) -> None:
    import logging

    from convmerge.cli import _StderrHandler, main

    # Simulate a plain CLI process: pytest installs its own root handlers.
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    lib_logger = logging.getLogger("convmerge")
    monkeypatch.setattr(lib_logger, "handlers", [])
    monkeypatch.setattr(lib_logger, "level", logging.NOTSET)

    src = _write(
        tmp_path / "in.jsonl", '{"text": "t", "instruction": "only instruction, no output"}\n'
    )
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
        ]
    )
    err = capsys.readouterr().err
    assert "warning: chat adapter: routing record to the 'text' branch" in err
    assert any(isinstance(h, _StderrHandler) for h in lib_logger.handlers)


def _write_bytes(p: Path, data: bytes) -> Path:
    p.write_bytes(data)
    return p


def test_iter_jsonl_skips_undecodable_bytes(tmp_path: Path) -> None:
    p = _write_bytes(tmp_path / "a.jsonl", b'{"a": "\xff\xfe"}\n{"a": "ok"}\n')
    seen: list[JsonlDecodeError] = []
    stats = ReadStats()
    lines = list(iter_jsonl(p, stats=stats, on_invalid=seen.append))
    assert [x.value for x in lines] == [{"a": "ok"}]
    assert (stats.invalid_json, stats.first_invalid_line) == (1, 1)
    assert "invalid utf-8 bytes" in str(seen[0])


def test_iter_jsonl_skips_unpaired_surrogate_but_keeps_pairs(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "a.jsonl",
        '{"a": "\\ud800"}\n{"a": "\\ud83d\\ude00"}\n{"a": "\\\\ud800"}\n',
    )
    seen: list[JsonlDecodeError] = []
    lines = list(iter_jsonl(p, on_invalid=seen.append))
    assert [x.value for x in lines] == [{"a": "\U0001f600"}, {"a": "\\ud800"}]
    assert "unpaired surrogate" in str(seen[0])
    for line in lines:
        line.raw.encode("utf-8")


def test_iter_jsonl_skips_too_deep_nesting(tmp_path: Path) -> None:
    p = _write(tmp_path / "a.jsonl", "[" * 100_000 + "]" * 100_000 + '\n{"a": 1}\n')
    stats = ReadStats()
    assert [x.value for x in iter_jsonl(p, stats=stats)] == [{"a": 1}]
    assert stats.invalid_json == 1
    with pytest.raises(JsonlDecodeError, match="nested too deeply"):
        list(iter_jsonl(p, on_error="raise"))


@pytest.mark.parametrize(
    "bad",
    [b'{"instruction": "\xff", "output": "a"}\n', b'{"instruction": "\\ud800", "output": "a"}\n'],
)
def test_commands_skip_bad_encoding_lines(tmp_path: Path, bad: bytes) -> None:
    from convmerge.convert import ConvertStats, convert_file
    from convmerge.normalize.dedup import deduplicate_jsonl
    from convmerge.split import split_jsonl

    good = b'{"instruction": "q", "output": "a"}\n'
    src = _write_bytes(tmp_path / "in.jsonl", bad + good)
    stats = ConvertStats()
    convert_file(src, tmp_path / "c.jsonl", adapter_name="alpaca", output_format="messages",
                 stats=stats)  # fmt: skip
    assert (stats.written, stats.invalid_json) == (1, 1)
    assert deduplicate_jsonl(src, tmp_path / "d.jsonl")[1] == 1
    split_jsonl(src, train_out=tmp_path / "t.jsonl", val_out=tmp_path / "v.jsonl", val=0.5)
    out = (tmp_path / "t.jsonl").read_bytes() + (tmp_path / "v.jsonl").read_bytes()
    assert out == good


def test_mix_skips_bad_encoding_lines(tmp_path: Path) -> None:
    from convmerge.mix import MixSource, mix_files

    src = _write_bytes(tmp_path / "in.jsonl", b'{"a": "\xff"}\n{"a": 1}\n{"a": 2}\n')
    out = tmp_path / "o.jsonl"
    mix_files([MixSource(path=src, weight=1.0)], out, total=2, seed=0)
    assert sorted(out.read_text(encoding="utf-8").splitlines()) == ['{"a": 1}', '{"a": 2}']


def test_parse_line_agrees_with_iter_jsonl(tmp_path: Path) -> None:
    from convmerge.io import _parse_line

    lines = [b'{"a": 1}', b"{bad", b'{"a": "\xff"}', b'{"a": "\\ud800"}',
             b'{"a": "\\ud83d\\ude00"}', b"[" * 100_000 + b"]" * 100_000, b'"s"']  # fmt: skip
    p = _write_bytes(tmp_path / "a.jsonl", b"\n".join(lines) + b"\n")
    good = {x.number: x.value for x in iter_jsonl(p)}
    raw_lines = p.read_bytes().decode("utf-8", "surrogateescape").splitlines()
    for number, raw in enumerate(raw_lines, 1):
        value, error = _parse_line(raw, check_bytes=True, encoding="utf-8")
        assert (error is None) == (number in good), number
        if error is None:
            assert value == good[number]


def test_other_input_encodings_are_written_as_utf8(tmp_path: Path) -> None:
    # ``encoding`` is the input's; every output file is UTF-8 (a cp949 file
    # used to come out as cp949).
    from convmerge.convert import convert_file
    from convmerge.mix import MixSource, mix_files
    from convmerge.quality import FilterSpec, filter_jsonl
    from convmerge.split import split_jsonl

    rows = [{"instruction": f"안녕하세요 질문 {i}", "output": f"답변입니다 {i}"} for i in range(4)]
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    src = _write_bytes(tmp_path / "in.jsonl", text.encode("cp949"))
    for workers in (1, 2):
        out = tmp_path / f"c{workers}.jsonl"
        convert_file(src, out, adapter_name="alpaca", output_format="messages",
                     encoding="cp949", workers=workers)  # fmt: skip
        assert "안녕하세요 질문 0" in out.read_text(encoding="utf-8")
    filter_jsonl(src, spec=FilterSpec(), output=tmp_path / "f.jsonl", encoding="cp949")
    split_jsonl(src, tmp_path / "t.jsonl", tmp_path / "v.jsonl", val_rows=1, encoding="cp949")
    mix_files([MixSource(path=src, weight=1.0)], tmp_path / "m.jsonl", total=4, seed=0,
              encoding="cp949")  # fmt: skip
    for name in ("f.jsonl", "t.jsonl", "m.jsonl"):
        assert (tmp_path / name).read_text(encoding="utf-8").count("\n") in (3, 4), name


def test_cli_suggests_encoding_for_undecodable_input(tmp_path: Path, capsys) -> None:
    from convmerge.cli import main

    text = json.dumps({"instruction": "질문", "output": "답변"}, ensure_ascii=False) + "\n"
    src = _write_bytes(tmp_path / "in.jsonl", text.encode("cp949"))
    main(["convert", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--from", "auto"])
    err = capsys.readouterr().err
    assert "pass --encoding NAME" in err and "normalize" not in err
    main(["dedupe", "-i", str(src), "-o", str(tmp_path / "d.jsonl")])
    assert "--encoding NAME" in capsys.readouterr().err
    broken = _write(tmp_path / "b.jsonl", '{"instruction": "q", "output": "a"}\n{bad\n')
    main(["convert", "-i", str(broken), "-o", str(tmp_path / "o2.jsonl"), "--from", "auto"])
    err = capsys.readouterr().err
    assert "normalize" in err and "--encoding" not in err


@pytest.mark.parametrize(
    "raw",
    ['{"a": [1, 2.5, null, true]}', ' {"a": 1}', '{"a": 1} ', '{"a": 1} x', "﻿{}",
     "{bad", "", '"\\ud800"', "[" * 100_000 + "]" * 100_000, '{"a": 1}{"b": 2}'],
)  # fmt: skip
def test_fast_loads_matches_json_loads(raw: str) -> None:
    from convmerge.io import _loads

    try:
        expected = json.loads(raw)
    except (ValueError, RecursionError) as e:
        with pytest.raises(type(e)) as got:
            _loads(raw)
        assert str(got.value) == str(e)
    else:
        assert _loads(raw) == expected
