"""scripts/quality.py runs the quality commands on catalog records (offline here)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
pytest.importorskip("datasketch")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_check_and_render(tmp_path: Path, monkeypatch) -> None:
    from convmerge.decontam import EvalIndex

    quality = _load("_quality_script", ROOT / "scripts" / "quality.py")
    catalog = quality._catalog_module()
    monkeypatch.setattr(catalog, "hub_rows", lambda entry, rows: [entry["record"]] * 3)
    index = EvalIndex()
    index.add("hf:x/eval:test", ["nothing in the catalog records says exactly this long line"])
    entries = {e["id"]: e for e in catalog.load_catalog()}
    results = [
        quality.check(entries[rid], 3, index, catalog)
        for rid in ("beomi/KoAlpaca-v1.1a", "HuggingFaceH4/ultrafeedback_binarized")
    ]
    ko, pref = results
    assert ko["rows"] == 3 and "script" in ko["rules"] and ko["near_duplicates"] == 2
    assert "preference" in pref and pref["decontam"] == {}
    text = quality.render([*results, {"id": "broken/x", "error": "boom"}], 3)
    assert "| beomi/KoAlpaca-v1.1a | 3 |" in text and "error: boom" in text
    json.dumps(results)
