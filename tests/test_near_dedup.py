"""convmerge dedupe --near: MinHash LSH near-duplicate removal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.normalize.near_dedup import NearDedupeStats, deduplicate_near_jsonl
from convmerge.recipe import RecipeError, parse_recipe, run

pytest.importorskip("datasketch")

BASE = (
    "The mitochondria is the powerhouse of the cell. It produces energy in the form of ATP "
    "through a process called cellular respiration, which takes place in several stages "
    "including glycolysis, the citric acid cycle, and oxidative phosphorylation. Glycolysis "
    "happens in the cytoplasm and splits glucose into two molecules of pyruvate, yielding a "
    "small amount of ATP and NADH. The pyruvate then enters the mitochondrial matrix, where "
    "the citric acid cycle oxidizes it to carbon dioxide while reducing more NAD+ and FAD. "
    "Finally, the electron transport chain in the inner membrane uses those electrons to pump "
    "protons, and ATP synthase lets them flow back to make most of the cell's ATP."
)


def _chat(user: str, answer: str) -> dict:
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": answer}]}  # fmt: skip


def _write(path: Path, rows: list) -> Path:
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows),
                    encoding="utf-8")  # fmt: skip
    return path


ROWS = [
    _chat("What does the mitochondria do?", BASE),
    # the same answer, lightly edited
    _chat("What does the mitochondria do?", BASE.replace("several stages", "a few stages")),
    # the same text in another layout
    {"instruction": "What does the mitochondria do?", "input": "", "output": BASE},
    _chat(
        "Explain photosynthesis.",
        "Plants turn light, water and carbon dioxide into "
        "glucose and oxygen inside their chloroplasts using chlorophyll.",
    ),  # fmt: skip
]


def test_near_duplicates(tmp_path: Path) -> None:
    st = NearDedupeStats()
    total, kept = deduplicate_near_jsonl(
        _write(tmp_path / "in.jsonl", [*ROWS, "{bad"]), tmp_path / "out.jsonl",
        rejects=tmp_path / "rej.jsonl", stats=st,
    )  # fmt: skip
    assert (total, kept, st.near_duplicates, st.invalid_json) == (5, 2, 2, 1)
    out = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert out == [ROWS[0], ROWS[3]]
    assert len((tmp_path / "rej.jsonl").read_text().splitlines()) == 2


def test_threshold_and_keys(tmp_path: Path) -> None:
    src = _write(tmp_path / "in.jsonl", ROWS)
    _, strict = deduplicate_near_jsonl(src, tmp_path / "o.jsonl", threshold=0.95)
    assert strict == 3  # only the exact text copy goes
    rows = [{"id": 1, "text": BASE}, {"id": 2, "text": BASE}]
    _, kept = deduplicate_near_jsonl(_write(tmp_path / "k.jsonl", rows), tmp_path / "o.jsonl",
                                     keys=["id"])  # fmt: skip
    assert kept == 2
    with pytest.raises(ValueError, match="threshold"):
        deduplicate_near_jsonl(src, tmp_path / "o.jsonl", threshold=0)
    with pytest.raises(ValueError, match="raise num_perm"):
        deduplicate_near_jsonl(src, tmp_path / "o.jsonl", threshold=0.99)


def test_cli(tmp_path: Path, capsys) -> None:
    src = _write(tmp_path / "in.jsonl", ROWS)
    main(["dedupe", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--near"])
    assert "near_duplicates=2" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["dedupe", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--near",
              "--threshold", "2"])  # fmt: skip
    assert e.value.code == 2


def test_recipe(tmp_path: Path) -> None:
    _write(tmp_path / "a.jsonl", ROWS)
    raw = {
        "output": "out.jsonl",
        "sources": {"a": {"path": "a.jsonl", "convert": {"from": "auto"}}},
        "dedupe": {"near": True, "threshold": 0.8},
    }
    result = run(parse_recipe(raw, path=tmp_path / "r.yaml"), log=lambda _m: None)
    assert result.report["steps"]["dedupe"]["stats"]["near_duplicates"] == 2
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 2
    with pytest.raises(RecipeError, match="dedupe.threshold"):
        parse_recipe({**raw, "dedupe": {"near": True, "threshold": 0}}, path=tmp_path / "r.yaml")
    with pytest.raises(RecipeError, match="dedupe.near"):
        parse_recipe({**raw, "dedupe": {"near": "yes"}}, path=tmp_path / "r.yaml")
