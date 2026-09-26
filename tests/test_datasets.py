"""Every dataset in tests/datasets/catalog.json converts, and the output is pinned.

Records reproduce each dataset's real layout with made-up content (the data
itself is not redistributed). ``python scripts/datasets.py check`` runs the
same conversions on real rows from the Hub.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from convmerge import ConvertStats, build_convert_config, convert_with_config

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent / "datasets"
EXPECTED = HERE / "expected.jsonl"
UPDATE = os.environ.get("CONVMERGE_UPDATE_GOLDEN") == "1"

_spec = importlib.util.spec_from_file_location("catalog_script", ROOT / "scripts" / "datasets.py")
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)
CATALOG = script.load_catalog()


def _convert(
    tmp_path: Path, record: dict, fmt: str, preference: str | None
) -> tuple[list, ConvertStats]:
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = build_convert_config(adapter="auto", output_format=fmt, preference=preference)
    stats = ConvertStats()
    convert_with_config(src, dst, cfg, stats=stats)
    return [json.loads(x) for x in dst.read_text(encoding="utf-8").splitlines()], stats


def _outputs(tmp_path: Path) -> dict[str, dict]:
    out = {}
    for e in CATALOG:
        rows, stats = _convert(tmp_path, e["record"], e["format"], e.get("preference"))
        assert stats.written == 1 and stats.dropped == 0 and stats.skipped == 0, (
            e["id"],
            stats.drop_reasons,
        )
        out[e["id"]] = rows[0]
    return out


def test_catalog_is_well_formed() -> None:
    ids = [e["id"] for e in CATALOG]
    assert len(ids) == len(set(ids)) and len(ids) >= 30
    assert {e["kind"] for e in CATALOG} == set(script.KINDS)


def test_every_catalog_dataset_converts_as_pinned(tmp_path: Path) -> None:
    got = _outputs(tmp_path)
    if UPDATE:
        EXPECTED.write_text(
            "".join(json.dumps({"id": k, "output": v}, ensure_ascii=False) + "\n"
                    for k, v in got.items()),
            encoding="utf-8",
        )  # fmt: skip
        return
    expected = {
        row["id"]: row["output"]
        for row in map(json.loads, EXPECTED.read_text(encoding="utf-8").splitlines())
    }
    assert got.keys() == expected.keys()
    for key in got:
        assert got[key] == expected[key], key


@pytest.mark.parametrize(
    "entry", [e for e in CATALOG if e["kind"] == "preference"], ids=lambda e: e["id"]
)
def test_preference_datasets_also_work_as_sft(tmp_path: Path, entry: dict) -> None:
    rows, stats = _convert(tmp_path, entry["record"], "messages", "chosen")
    assert stats.written == 1, stats.drop_reasons
    assert rows[0]["messages"][-1]["role"] == "assistant"


def test_readme_table_matches_catalog() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = script.render_table(CATALOG)
    assert f"{script.START}\n{table}\n{script.END}" in readme, (
        "README dataset table is stale: run `python scripts/datasets.py table --write`"
    )
