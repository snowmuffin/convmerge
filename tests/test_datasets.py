"""Every dataset in tests/datasets/catalog.json converts, and the output is pinned.

Records reproduce each dataset's real layout with made-up content (the data
itself is not redistributed). ``python scripts/datasets.py check`` runs the
same conversions on real rows from the Hub.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from convmerge import ConvertStats, convert_with_config

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent / "datasets"
EXPECTED = HERE / "expected.jsonl"
UPDATE = os.environ.get("CONVMERGE_UPDATE_GOLDEN") == "1"

_spec = importlib.util.spec_from_file_location("catalog_script", ROOT / "scripts" / "datasets.py")
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = script  # the script defines dataclasses
_spec.loader.exec_module(script)
CATALOG = script.load_catalog()


def _convert(tmp_path: Path, entry: dict) -> tuple[list, ConvertStats]:
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in script.sample_rows(entry)),
                   encoding="utf-8")  # fmt: skip
    cfg = script.convert_config(entry)
    stats = ConvertStats()
    convert_with_config(src, dst, cfg, stats=stats)
    return [json.loads(x) for x in dst.read_text(encoding="utf-8").splitlines()], stats


def _outputs(tmp_path: Path) -> dict[str, dict]:
    out = {}
    for e in CATALOG:
        rows, stats = _convert(tmp_path, e)
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
    rows, stats = _convert(tmp_path, {**entry, "format": "messages", "preference": "chosen"})
    assert stats.written == 1, stats.drop_reasons
    assert rows[0]["messages"][-1]["role"] == "assistant"


@pytest.mark.parametrize(
    "entry", [e for e in CATALOG if e["kind"] == "reasoning"], ids=lambda e: e["id"]
)
def test_reasoning_datasets_keep_their_trace(tmp_path: Path, entry: dict) -> None:
    for mode, key in (("reasoning_content", "reasoning_content"), ("thinking", "thinking")):
        rows, _ = _convert(tmp_path, {**entry, "emit": {"reasoning": mode}})
        answer = rows[0]["messages"][-1]
        assert answer[key] and "<think>" not in answer["content"], (entry["id"], answer)


def test_readme_table_matches_catalog() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = script.render_table(CATALOG)
    assert f"{script.START}\n{table}\n{script.END}" in readme, (
        "README dataset table is stale: run `python scripts/datasets.py table --write`"
    )


def _catalog_rows(entry: dict, rows: int) -> list[dict]:
    """A stand-in for the Hub: the catalog record, repeated."""
    return script.sample_rows(entry) * min(rows, 3)


def test_live_check_logic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    summary = tmp_path / "summary.md"
    assert script.check(5, None, summary=summary, loader=_catalog_rows) == 0
    text = summary.read_text(encoding="utf-8")
    assert "| ✅ | tatsu-lab/alpaca | 3/3 |" in text
    assert "| ⏭️ | lmsys/lmsys-chat-1m |" in text and "gated: set HF_TOKEN" in text
    assert "| ✅ | simplescaling/s1K-1.1 | 3/3 | 3 |" in text

    reasoning = next(e for e in CATALOG if e["id"] == "simplescaling/s1K-1.1")
    no_trace = {**reasoning, "adapter_kwargs": None}  # the trace column is not mapped
    r = script.check_entry(no_trace, 5, loader=_catalog_rows)
    assert (r.status, r.written) == ("fail", 3) and "3 rows carry a reasoning trace" in r.note

    # Rows without any trace (e.g. Llama-Nemotron "reasoning: off" rows) only warn.
    nemotron = next(e for e in CATALOG if "Nemotron" in e["id"])
    plain = {**nemotron["record"], "output": "Two.", "reasoning": "off"}
    r = script.check_entry(nemotron, 5, loader=lambda e, n: [plain] * 3)
    assert r.status == "warn" and "no reasoning trace" in r.note
    assert not script.has_trace(plain) and script.has_trace(nemotron["record"])

    alpaca = next(e for e in CATALOG if e["id"] == "tatsu-lab/alpaca")

    def half_bad(entry: dict, rows: int) -> list[dict]:
        return [entry["record"], {"unrelated": 1}]

    assert script.check_entry(alpaca, 5, loader=half_bad).status == "fail"
    assert script.check_entry(alpaca, 5, loader=half_bad, min_ok=0.5).status == "warn"

    def broken(entry: dict, rows: int) -> list[dict]:
        raise ConnectionError("hub down | retry")

    r = script.check_entry(alpaca, 5, loader=broken)
    assert r.status == "fail" and "ConnectionError" in r.note
    assert "hub down \\| retry" in script.render_summary([r], 5)


def test_check_script_imports_the_datasets_library(tmp_path: Path) -> None:
    """Run as ``python scripts/datasets.py``, the script must not import itself as ``datasets``."""
    import subprocess

    fake = tmp_path / "fake" / "datasets"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(
        "class _Rows:\n"
        "    def take(self, n):\n"
        "        return [{'instruction': 'Say hi.', 'input': '', 'output': 'Hi!'}] * n\n"
        "def load_dataset(*args, **kwargs):\n"
        "    return _Rows()\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(fake.parent), str(ROOT / "src")]),
    }
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "datasets.py"), "check",
         "--only", "tatsu-lab/alpaca", "--rows", "3"],
        capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok    tatsu-lab/alpaca: 3/3" in proc.stdout


def test_check_entry_shows_dropped_rows(capsys) -> None:
    alpaca = next(e for e in CATALOG if e["id"] == "tatsu-lab/alpaca")
    rows = [alpaca["record"], {"instruction": "q", "output": ""}]
    script.check_entry(alpaca, 2, loader=lambda e, n: rows, show_drops=5)
    out = capsys.readouterr().out
    assert "--- tatsu-lab/alpaca line 2: no_assistant" in out
    assert '"instruction": "q"' in out
