"""Score columns: convert --keep-fields carries them into meta, filter meta_min drops by them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.config import build_convert_config
from convmerge.convert import convert_file, convert_records
from convmerge.emitters import EmitOptions
from convmerge.quality import FilterSpec, FilterStats, filter_jsonl, load_rules_file
from convmerge.recipe import RecipeError, parse_recipe, run


def _write(path: Path, rows: list) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _rated(i: int, helpfulness: object, correctness: object = 4) -> dict:
    return {"prompt": f"q{i}", "response": f"a{i}", "helpfulness": helpfulness,
            "correctness": correctness, "verbosity": 2}  # fmt: skip


KEEP = {"from": "auto", "keep_fields": ["helpfulness"]}


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_keep_fields_serial_workers_and_records(tmp_path: Path, monkeypatch) -> None:
    from convmerge import _parallel

    monkeypatch.setattr(_parallel, "CHUNK_LINES", 2)
    src = _write(tmp_path / "in.jsonl", [_rated(i, i % 5) for i in range(7)])
    opts = EmitOptions(keep_fields=["helpfulness", "correctness", "absent"])
    outs = []
    for workers in (1, 3):
        out = tmp_path / f"o{workers}.jsonl"
        convert_file(src, out, adapter_name="auto", output_format="messages",
                     emit_options=opts, workers=workers)  # fmt: skip
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    rows = _read(tmp_path / "o1.jsonl")
    assert [r["meta"] for r in rows[:2]] == [
        {"helpfulness": 0, "correctness": 4}, {"helpfulness": 1, "correctness": 4},
    ]  # fmt: skip
    records = [_rated(0, 3)]
    assert next(convert_records(records, emit_options=opts))["meta"] == {
        "helpfulness": 3, "correctness": 4,
    }  # fmt: skip
    assert "meta" not in next(convert_records(records))  # off by default
    # Nothing to copy: no meta key is added.
    assert "meta" not in next(convert_records(records, emit_options=EmitOptions(
        keep_fields=["absent"])))  # fmt: skip


def test_keep_fields_meta_key_and_keep_meta(tmp_path: Path) -> None:
    rec = {"id": "r1", "score": 0.5, "messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}  # fmt: skip
    opts = EmitOptions(keep_meta=["id"], keep_fields=["score"], meta_key="_m")
    row = next(convert_records([rec], emit_options=opts))
    assert row["_m"] == {"id": "r1", "score": 0.5}
    assert EmitOptions(keep_fields=["a", "b", "a"]).keep_fields == ("a", "b")
    for bad in ("helpfulness", [""], [1]):
        with pytest.raises(ValueError, match="keep_fields"):
            EmitOptions(keep_fields=bad)  # type: ignore[arg-type]


def test_keep_fields_cli_preset_and_recipe(tmp_path: Path) -> None:
    src = _write(tmp_path / "in.jsonl", [_rated(0, 2), _rated(1, 4)])
    out = tmp_path / "o.jsonl"
    base = ["convert", "-i", str(src), "-o", str(out), "--from", "auto"]
    main([*base, "--keep-fields", "helpfulness, correctness"])
    assert [r["meta"]["helpfulness"] for r in _read(out)] == [2, 4]
    with pytest.raises(SystemExit) as e:
        main([*base, "--keep-fields", " , "])
    assert e.value.code == 2

    preset = tmp_path / "p.json"
    preset.write_text(json.dumps({"adapter": "auto", "output_format": "messages",
                                  "output_options": {"keep_fields": ["verbosity"]}}))  # fmt: skip
    cfg = build_convert_config(preset_path=preset)
    assert cfg.emit_options.keep_fields == ("verbosity",)

    raw = {
        "output": "out.jsonl",
        "sources": {"a": {"path": "in.jsonl", "convert": KEEP}},
    }
    run(parse_recipe(raw, path=tmp_path / "recipe.yaml"), log=lambda _m: None)
    assert [r["meta"] for r in _read(tmp_path / "out.jsonl")] == [
        {"helpfulness": 2}, {"helpfulness": 4},
    ]  # fmt: skip
    for bad in ("helpfulness", [], [3]):
        convert = {**KEEP, "keep_fields": bad}
        bad_raw = {**raw, "sources": {"a": {"path": "in.jsonl", "convert": convert}}}
        with pytest.raises(RecipeError, match="keep_fields"):
            parse_recipe(bad_raw, path=tmp_path / "r.yaml")


def _scored(*scores: object, key: str = "meta") -> list:
    rows: list = []
    for i, s in enumerate(scores):
        row: dict = {"messages": [{"role": "user", "content": f"q{i}"},
                                  {"role": "assistant", "content": f"answer {i}"}]}  # fmt: skip
        if s is not None:
            row[key] = {"helpfulness": s} if s != "nometa" else "not a dict"
        rows.append(row)
    return rows


def test_meta_min_rule(tmp_path: Path) -> None:
    rows = _scored(4, 3, 2.5, None, "nometa", "3", True, float("nan"))
    src = tmp_path / "in.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    spec = FilterSpec.from_options(meta_min={"helpfulness": 3})
    assert "meta_min" in spec.rules
    st = filter_jsonl(src, spec=spec, output=tmp_path / "o.jsonl")
    assert (st.kept, st.rules["meta_min"]) == (2, 6)
    assert [s["text"] for s in st.samples["meta_min"]] == [
        "helpfulness=2.5 (min 3)", "helpfulness=missing (min 3)", "helpfulness=missing (min 3)",
    ]  # fmt: skip
    # Off unless asked for, and the default rules still run alongside it.
    assert "meta_min" not in filter_jsonl(src).rules
    assert FilterSpec.from_options(meta_min={"helpfulness": 3},
                                   disable=["meta_min"]).rules.count("meta_min") == 0  # fmt: skip

    other = _write(tmp_path / "k.jsonl", _scored(1, 5, key="_m"))
    spec = FilterSpec.from_options(meta_min={"helpfulness": 2}, meta_key="_m")
    assert filter_jsonl(other, spec=spec).kept == 1

    for bad, match in (
        ({"rules": ("meta_min",)}, "needs meta_min"),
        ({"meta_min": {"helpfulness": "3"}}, "meta_min.helpfulness"),
        ({"meta_min": {"helpfulness": True}}, "meta_min.helpfulness"),
        ({"meta_min": {"helpfulness": float("nan")}}, "meta_min.helpfulness"),
        ({"meta_min": {"": 1}}, "field names"),
        ({"meta_key": ""}, "meta_key"),
    ):
        with pytest.raises(ValueError, match=match):
            FilterSpec(**bad)  # type: ignore[arg-type]


def test_meta_min_rules_file_cli_and_workers(tmp_path: Path, capsys, monkeypatch) -> None:
    from convmerge import _parallel

    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"meta_min": {"helpfulness": 3, "correctness": 2}}))
    spec = FilterSpec.from_options(rules_file=rules)
    assert spec.meta_min == {"helpfulness": 3, "correctness": 2} and "meta_min" in spec.rules
    # A minimum given directly wins over the file's, field by field.
    spec = FilterSpec.from_options(rules_file=rules, meta_min={"helpfulness": 1})
    assert spec.meta_min == {"helpfulness": 1, "correctness": 2}
    for bad in ({"meta_min": []}, {"meta_min": {}}, {"meta_min": {"h": "x"}}):
        rules.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="meta_min"):
            load_rules_file(rules)

    src = _write(tmp_path / "in.jsonl", _scored(*[i % 5 for i in range(10)]))
    main(["filter", "-i", str(src), "-o", str(tmp_path / "o.jsonl"),
          "--meta-min", "helpfulness=3"])  # fmt: skip
    report = json.loads(capsys.readouterr().out)
    assert (report["kept"], report["rules"]["meta_min"]) == (4, 6)
    for bad in ("helpfulness", "helpfulness=x", "=3", "helpfulness=nan"):
        with pytest.raises(SystemExit) as e:
            main(["filter", "-i", str(src), "--meta-min", bad])
        assert e.value.code == 2

    monkeypatch.setattr(_parallel, "CHUNK_LINES", 3)
    spec = FilterSpec.from_options(meta_min={"helpfulness": 2})
    results = []
    for workers in (1, 3):
        st = FilterStats()
        out = tmp_path / f"w{workers}.jsonl"
        filter_jsonl(src, spec=spec, output=out, stats=st, workers=workers)
        results.append((out.read_bytes(), st.to_report()))
    assert results[0] == results[1]


def test_meta_min_recipe(tmp_path: Path) -> None:
    _write(tmp_path / "in.jsonl", [_rated(0, 1), _rated(1, 3), _rated(2, 4)])
    raw = {
        "output": "out.jsonl",
        "sources": {"a": {"path": "in.jsonl", "convert": KEEP}},
        "filter": {"meta_min": {"helpfulness": 3}},
    }
    result = run(parse_recipe(raw, path=tmp_path / "recipe.yaml"), log=lambda _m: None)
    assert result.report["steps"]["filter"]["stats"]["rules"]["meta_min"] == 1
    assert [r["meta"]["helpfulness"] for r in _read(tmp_path / "out.jsonl")] == [3, 4]
    for bad, match in (
        ({"meta_min": {"helpfulness": "3"}}, "meta_min"),
        ({"meta_min": [3]}, "filter.meta_min"),
        ({"meta_key": 1}, "filter.meta_key"),
    ):
        with pytest.raises(RecipeError, match=match):
            parse_recipe({**raw, "filter": bad}, path=tmp_path / "r.yaml")
