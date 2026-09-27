"""convmerge split: content-hashed train/validation split."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from convmerge import SplitStats, split_jsonl
from convmerge.cli import main
from convmerge.recipe import load_lock, parse_recipe, run


def _rows(n: int, prefix: str = "q") -> list[dict]:
    return [
        {"messages": [{"role": "user", "content": f"{prefix}{i}"},
                      {"role": "assistant", "content": f"a{i}"}]}
        for i in range(n)
    ]  # fmt: skip


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_ratio_split_is_complete_disjoint_and_about_right(tmp_path: Path) -> None:
    src = _write(tmp_path / "all.jsonl", _rows(2000))
    stats = SplitStats()
    train, val = split_jsonl(src, tmp_path / "t.jsonl", tmp_path / "v.jsonl", val=0.1, stats=stats)
    t, v = _lines(tmp_path / "t.jsonl"), _lines(tmp_path / "v.jsonl")
    assert (train, val) == (len(t), len(v)) == (stats.train, stats.val)
    assert sorted(t + v) == sorted(_lines(src)) and not set(t) & set(v)
    assert 150 < val < 250  # ~10% of 2000


def test_assignment_depends_on_content_not_order(tmp_path: Path) -> None:
    rows = _rows(500)
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    a = _write(tmp_path / "a.jsonl", rows)
    b = _write(tmp_path / "b.jsonl", shuffled + _rows(50, prefix="new"))
    split_jsonl(a, tmp_path / "ta", tmp_path / "va", val=0.2, seed=7)
    split_jsonl(b, tmp_path / "tb", tmp_path / "vb", val=0.2, seed=7)
    val_a, val_b = set(_lines(tmp_path / "va")), set(_lines(tmp_path / "vb"))
    assert val_a == {line for line in val_b if '"new' not in line}
    split_jsonl(a, tmp_path / "tc", tmp_path / "vc", val=0.2, seed=8)
    assert set(_lines(tmp_path / "vc")) != val_a  # the seed matters


def test_duplicates_and_key_groups_stay_together(tmp_path: Path) -> None:
    rows = _rows(300)
    src = _write(tmp_path / "dup.jsonl", rows + rows)  # every row twice
    split_jsonl(src, tmp_path / "t", tmp_path / "v", val=0.3)
    t, v = _lines(tmp_path / "t"), _lines(tmp_path / "v")
    assert not set(t) & set(v) and len(v) % 2 == 0

    grouped = [{"prompt": f"p{i // 3}", "answer": i} for i in range(300)]
    src = _write(tmp_path / "g.jsonl", grouped)
    split_jsonl(src, tmp_path / "gt", tmp_path / "gv", val=0.3, keys=["prompt"])
    prompts = lambda p: {json.loads(x)["prompt"] for x in _lines(p)}  # noqa: E731
    assert not prompts(tmp_path / "gt") & prompts(tmp_path / "gv")


def test_exact_val_rows(tmp_path: Path) -> None:
    src = _write(tmp_path / "all.jsonl", _rows(1000))
    assert split_jsonl(src, tmp_path / "t", tmp_path / "v", val_rows=37) == (963, 37)
    first = set(_lines(tmp_path / "v"))
    shuffled = _lines(src)
    random.Random(3).shuffle(shuffled)
    (tmp_path / "s.jsonl").write_text("\n".join(shuffled) + "\n")
    split_jsonl(tmp_path / "s.jsonl", tmp_path / "t2", tmp_path / "v2", val_rows=37)
    assert set(_lines(tmp_path / "v2")) == first  # still content-based
    assert split_jsonl(src, tmp_path / "t3", tmp_path / "v3", val_rows=0) == (1000, 0)
    assert split_jsonl(src, tmp_path / "t4", tmp_path / "v4", val_rows=5000) == (0, 1000)


def test_invalid_lines_are_dropped_and_counted(tmp_path: Path) -> None:
    src = tmp_path / "bad.jsonl"
    src.write_text('{"a": 1}\n\n{broken\n{"a": 2}\n', encoding="utf-8")
    stats = SplitStats()
    split_jsonl(src, tmp_path / "t", tmp_path / "v", val=0.5, stats=stats)
    assert stats.train + stats.val == 2 and stats.invalid_json == 1
    assert stats.first_invalid_line == 3 and stats.total == 3


@pytest.mark.parametrize(
    ("kw", "message"),
    [({}, "exactly one"), ({"val": 0.1, "val_rows": 3}, "exactly one"), ({"val": 1.0}, "between")],
)
def test_bad_arguments(tmp_path: Path, kw: dict, message: str) -> None:
    src = _write(tmp_path / "a.jsonl", _rows(3))
    with pytest.raises(ValueError, match=message):
        split_jsonl(src, tmp_path / "t", tmp_path / "v", **kw)


def test_cli(tmp_path: Path, capsys) -> None:
    src = _write(tmp_path / "all.jsonl", _rows(100))
    main(["split", "-i", str(src), "-o", str(tmp_path / "train.jsonl"), "--val-rows", "10"])
    assert len(_lines(tmp_path / "train.val.jsonl")) == 10
    assert "val=10" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["split", "-i", str(src), "-o", str(tmp_path / "x.jsonl"), "--val", "1.5"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["split", "-i", str(tmp_path / "missing"), "-o", str(tmp_path / "x"), "--val", "0.1"])
    assert exc.value.code == 1


def test_recipe_split_step(tmp_path: Path) -> None:
    _write(tmp_path / "a.jsonl", _rows(200))
    raw = {
        "output": "out/train.jsonl",
        "sources": {"a": {"path": "a.jsonl", "normalize": False, "convert": {"from": "auto"}}},
        "dedupe": True,
        "split": {"val_rows": 20, "seed": 5},
    }
    recipe = parse_recipe(raw, path=tmp_path / "r.yaml")
    first = run(recipe, log=lambda _m: None)
    assert first.ran == ["a.convert", "dedupe", "split.train", "split.val"]
    assert len(_lines(tmp_path / "out/train.jsonl")) == 180
    assert len(_lines(tmp_path / "out/train.val.jsonl")) == 20
    assert set(load_lock(tmp_path / "r.lock.json")["steps"]) >= {"split.train", "split.val"}
    assert run(recipe, log=lambda _m: None).ran == []
    (tmp_path / "out/train.val.jsonl").unlink()  # only the missing part is rebuilt
    assert run(recipe, log=lambda _m: None).ran == ["split.val"]
    raw["split"]["val_output"] = "out/dev.jsonl"
    recipe = parse_recipe(raw, path=tmp_path / "r.yaml")
    assert run(recipe, log=lambda _m: None).ran == ["split.val"]
    assert len(_lines(tmp_path / "out/dev.jsonl")) == 20


@pytest.mark.parametrize(
    ("split", "message"),
    [
        ({}, "exactly one"),
        ({"val": 2}, "split.val"),
        ({"val_rows": -1}, "split.val_rows"),
        ({"val": 0.1, "val_output": "o.jsonl"}, "must differ"),
        ({"val": 0.1, "colour": 1}, "split.colour: unknown key"),
    ],
)
def test_recipe_split_schema_errors(tmp_path: Path, split: dict, message: str) -> None:
    from convmerge.recipe import RecipeError

    raw = {"output": "o.jsonl", "sources": {"a": {"path": "a", "convert": {"from": "auto"}}},
           "split": split}  # fmt: skip
    with pytest.raises(RecipeError, match=message):
        parse_recipe(raw, path=tmp_path / "r.yaml")
