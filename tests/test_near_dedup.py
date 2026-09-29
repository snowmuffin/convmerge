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
    main(["dedupe", "-i", str(src), "-o", str(tmp_path / "w.jsonl"), "--near", "--workers", "2"])
    assert (tmp_path / "w.jsonl").read_bytes() == (tmp_path / "o.jsonl").read_bytes()
    with pytest.raises(SystemExit) as e:
        main(["dedupe", "-i", str(src), "-o", str(tmp_path / "o.jsonl"), "--workers", "2"])
    assert e.value.code == 2
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
    for bad in ({"near": True, "workers": 0}, {"workers": 2}):
        with pytest.raises(RecipeError, match="dedupe.workers"):
            parse_recipe({**raw, "dedupe": bad}, path=tmp_path / "r.yaml")
    two = {**raw, "dedupe": {**raw["dedupe"], "workers": 2}}
    assert run(parse_recipe(two, path=tmp_path / "r.yaml"), log=lambda _m: None).ran == []


def test_shared_system_prompts_do_not_count(tmp_path: Path) -> None:
    system = {"role": "system", "content": BASE}
    rows = [{"messages": [system, {"role": "user", "content": q},
                          {"role": "assistant", "content": a}]}
            for q, a in (("What is 2+2?", "4"), ("Name a color.", "Blue"))]  # fmt: skip
    _, kept = deduplicate_near_jsonl(_write(tmp_path / "in.jsonl", rows), tmp_path / "o.jsonl")
    assert kept == 2


def _variants(n: int) -> list:
    """Rows in families of near-copies (random word edits), plus broken lines."""
    import random

    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(400)]
    rows: list = []
    for family in range(n):
        base = [rng.choice(vocab) for _ in range(rng.randint(3, 80))]
        for _ in range(rng.randint(1, 4)):
            text = list(base)
            for _ in range(rng.randint(0, 6)):
                text[rng.randrange(len(text))] = rng.choice(vocab)
            rows.append(_chat(f"q{family}", " ".join(text)))
        if family % 25 == 0:
            rows += ["{bad", ""]
    return rows


def _reference(rows: list, threshold: float, num_perm: int) -> list[str]:
    """What 1.0 kept: datasketch's own MinHashLSH over the same row text."""
    import datasketch as ds

    from convmerge.adapter_resolve import resolve_adapter
    from convmerge.normalize.near_dedup import _row_text, _shingles

    adapter = resolve_adapter("auto", None, pairs=True)
    lsh, kept = ds.MinHashLSH(threshold=threshold, num_perm=num_perm), []
    for i, row in enumerate(r for r in rows if isinstance(r, dict)):
        mh = ds.MinHash(num_perm=num_perm)
        mh.update_batch([s.encode() for s in _shingles(_row_text(row, adapter, None), 5)])
        if not lsh.query(mh):
            lsh.insert(i, mh)
            kept.append(json.dumps(row))
    return kept


@pytest.mark.parametrize(("threshold", "num_perm"), [(0.8, 128), (0.5, 64), (0.9, 256)])
def test_index_matches_minhash_lsh(tmp_path: Path, threshold: float, num_perm: int) -> None:
    rows = _variants(120)
    src = _write(tmp_path / "in.jsonl", rows)
    out = tmp_path / "out.jsonl"
    deduplicate_near_jsonl(src, out, threshold=threshold, num_perm=num_perm)
    kept = out.read_text().splitlines()
    assert kept == _reference(rows, threshold, num_perm)
    assert 0 < len(kept) < sum(isinstance(r, dict) for r in rows)


def test_workers_match_one_process(tmp_path: Path, monkeypatch) -> None:
    from convmerge import _parallel

    monkeypatch.setattr(_parallel, "CHUNK_LINES", 7)
    src = _write(tmp_path / "in.jsonl", _variants(60))
    with src.open("ab") as f:
        f.write(b'{"text": "\xff"}\n')
    results = []
    for workers in (1, 3):
        st = NearDedupeStats()
        out, rej = tmp_path / f"o{workers}.jsonl", tmp_path / f"r{workers}.jsonl"
        deduplicate_near_jsonl(src, out, rejects=rej, stats=st, workers=workers)
        results.append((out.read_bytes(), rej.read_bytes(), st))
    assert results[0] == results[1]
    assert results[1][2].invalid_json == 4 and results[1][2].first_invalid_line is not None
    with pytest.raises(ValueError, match="workers"):
        deduplicate_near_jsonl(src, tmp_path / "o.jsonl", workers=0)
