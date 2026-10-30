"""convmerge order: deterministic curriculum ordering, stages, CLI and recipe step."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.order import OrderStats, order_jsonl
from convmerge.recipe import RecipeError, parse_recipe, run


def _row(i: int, answer_words: int, turns: int = 1, **meta: object) -> dict:
    msgs = []
    for t in range(turns):
        msgs += [{"role": "user", "content": f"q{i}.{t}"},
                 {"role": "assistant", "content": " ".join(["w"] * answer_words)}]  # fmt: skip
    return {"messages": msgs, **({"meta": meta} if meta else {})}


def _write(path: Path, rows: list) -> Path:
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows),
                    encoding="utf-8")  # fmt: skip
    return path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_by_length_turns_and_meta(tmp_path: Path) -> None:
    rows = [_row(i, (i * 7) % 10 + 1, turns=i % 3 + 1, score=i % 4) for i in range(12)]
    src = _write(tmp_path / "in.jsonl", rows)

    order_jsonl(src, tmp_path / "len.jsonl")
    lengths = [sum(len(m["content"]) for m in r["messages"]) for r in _read(tmp_path / "len.jsonl")]
    assert lengths == sorted(lengths)

    order_jsonl(src, tmp_path / "turns.jsonl", by="turns", descending=True)
    turns = [len(r["messages"]) // 2 for r in _read(tmp_path / "turns.jsonl")]
    assert turns == sorted(turns, reverse=True)

    st = order_jsonl(src, tmp_path / "meta.jsonl", by="meta:score")
    scores = [r["meta"]["score"] for r in _read(tmp_path / "meta.jsonl")]
    assert scores == sorted(scores) and st.unranked == 0
    # Same rows in another order: the same bytes (ties go by a seeded hash).
    shuffled = rows[:]
    random.Random(3).shuffle(shuffled)
    order_jsonl(_write(tmp_path / "sh.jsonl", shuffled), tmp_path / "meta2.jsonl", by="meta:score")
    assert (tmp_path / "meta2.jsonl").read_bytes() == (tmp_path / "meta.jsonl").read_bytes()
    order_jsonl(src, tmp_path / "seed.jsonl", by="meta:score", seed=1)
    assert (tmp_path / "seed.jsonl").read_bytes() != (tmp_path / "meta.jsonl").read_bytes()


def test_stages_and_unranked(tmp_path: Path) -> None:
    rows = [_row(i, 1, score=i) for i in range(10)]
    rows += [_row(10, 1), _row(11, 1, score="high"), {"foo": 1}]
    src = _write(tmp_path / "in.jsonl", rows)
    st = OrderStats()
    order_jsonl(src, tmp_path / "o.jsonl", by="meta:score", stages=3, stats=st)
    out = _read(tmp_path / "o.jsonl")
    assert (st.rows, st.unranked) == (13, 3) and len(out) == 13
    assert [s["rows"] for s in st.stages] == [4, 4, 2]
    assert [(s["min"], s["max"]) for s in st.stages] == [(0, 3), (4, 7), (8, 9)]
    ranks = [r["meta"]["score"] for r in out[:10]]
    assert sorted(ranks[:4]) == [0, 1, 2, 3] and sorted(ranks[4:8]) == [4, 5, 6, 7]
    assert ranks != sorted(ranks)  # shuffled within stages
    assert {json.dumps(r) for r in out[10:]} == {json.dumps(r) for r in rows[10:]}
    st = order_jsonl(src, tmp_path / "d.jsonl", by="meta:score", descending=True, stages=2)
    assert [(s["min"], s["max"]) for s in st.stages] == [(5, 9), (0, 4)]

    for kwargs, match in (({"by": "colour"}, "order by"), ({"by": "meta:"}, "order by"),
                          ({"stages": 0}, "stages")):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            order_jsonl(src, tmp_path / "x.jsonl", **kwargs)  # type: ignore[arg-type]


def test_bytes_bom_and_crlf_kept(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    long_row, short_row = json.dumps(_row(0, 9), ensure_ascii=False), json.dumps(_row(1, 1))
    src.write_bytes(b"\xef\xbb\xbf" + long_row.encode() + b"\r\n" + short_row.encode() + b"\r\n")
    order_jsonl(src, tmp_path / "o.jsonl")
    assert (tmp_path / "o.jsonl").read_text(encoding="utf-8") == f"{short_row}\n{long_row}\n"


def test_cli_and_recipe(tmp_path: Path, capsys) -> None:
    src = _write(tmp_path / "a.jsonl", [_row(i, 10 - i) for i in range(6)])
    main(["order", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--stages", "2"])
    report = json.loads(capsys.readouterr().out)
    assert report["rows"] == 6 and len(report["stages"]) == 2
    with pytest.raises(SystemExit) as e:
        main(["order", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--by", "colour"])
    assert e.value.code == 2

    raw = {"output": "out.jsonl",
           "sources": {"a": {"path": "a.jsonl", "convert": {"from": "auto"}}},
           "order": {"by": "length", "stages": 2, "seed": 1}}  # fmt: skip
    recipe = parse_recipe(raw, path=tmp_path / "recipe.yaml")
    result = run(recipe, log=lambda _m: None)
    assert result.ran[-1] == "order"
    assert result.report["steps"]["order"]["stats"]["stages"][0]["rows"] == 3
    assert run(recipe, log=lambda _m: None).ran == []
    for bad, match in (({"by": "x"}, "order.by"), ({"stages": 0}, "order.stages"),
                       ({"descending": "yes"}, "order.descending"), ({"seed": 1.5}, "order.seed"),
                       ({"colour": 1}, "order.colour")):  # fmt: skip
        with pytest.raises(RecipeError, match=match):
            parse_recipe({**raw, "order": bad}, path=tmp_path / "r.yaml")
    assert parse_recipe({**raw, "order": True}, path=tmp_path / "r.yaml").order is not None
