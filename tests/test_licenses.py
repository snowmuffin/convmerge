"""License provenance: detection, warnings, the recipe report, and --meta."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from convmerge import EmitOptions
from convmerge.cli import main
from convmerge.emitters import emit_messages
from convmerge.licenses import detect_hf_license, license_summary, license_warning
from convmerge.models import ChatMessage, TrainingExample
from convmerge.recipe import RecipeError, parse_recipe, run


@pytest.mark.parametrize(
    ("license", "problem"),
    [
        ("apache-2.0", None),
        ("mit", None),
        ("cc-by-sa-4.0", None),
        ("cc-by-nc-4.0", "non-commercial"),
        ("CC-BY-NC-SA-4.0", "non-commercial"),
        ("research only", "non-commercial"),
        ("non-commercial", "non-commercial"),
        ("other", "custom terms"),
        (None, "no known license"),
        ("unknown", "no known license"),
    ],
)
def test_license_warning(license: str | None, problem: str | None) -> None:
    got = license_warning(license)
    assert (got is None) if problem is None else (got is not None and problem in got)


def test_detect_hf_license(monkeypatch) -> None:
    class Card(dict):
        pass

    class Api:
        def dataset_info(self, repo_id, token=None):
            if repo_id == "down/net":
                raise ConnectionError("offline")
            licenses = {"a/b": "mit", "c/d": ["cc-by-nc-4.0", "other"], "e/f": None}
            return types.SimpleNamespace(card_data=Card(license=licenses[repo_id]))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api))
    assert detect_hf_license("a/b") == "mit"
    assert detect_hf_license("c/d") == "cc-by-nc-4.0, other"
    assert detect_hf_license("e/f") is None
    assert detect_hf_license("down/net") is None


def test_summary_prefers_declared() -> None:
    summary, warnings = license_summary(
        {
            "kullm": {"declared": None, "detected": "apache-2.0", "rows": 10},
            "ko_uf": {"declared": "non-commercial (dataset card)", "detected": None, "rows": 5},
            "local": {"declared": None, "detected": None, "rows": 3},
        }
    )
    assert summary["kullm"] == {"license": "apache-2.0", "source": "card", "rows": 10}
    assert summary["ko_uf"]["source"] == "declared" and "warning" in summary["ko_uf"]
    assert summary["local"]["license"] is None
    assert warnings == [
        "ko_uf (non-commercial (dataset card)): non-commercial or research-only terms",
        "local: no known license",
    ]


def test_meta_values() -> None:
    ex = TrainingExample(
        messages=[ChatMessage("user", "q"), ChatMessage("assistant", "a")],
        meta={"source": "chat", "id": 3},
    )
    opts = EmitOptions(meta_values={"dataset": "kullm"})
    assert emit_messages(ex, options=opts)["meta"] == {"dataset": "kullm"}
    both = EmitOptions(keep_meta=("id",), meta_values={"dataset": "kullm", "id": "x"})
    assert emit_messages(ex, options=both)["meta"] == {"id": "x", "dataset": "kullm"}
    with pytest.raises(ValueError, match="meta_values"):
        EmitOptions(meta_values={"n": 1})  # type: ignore[dict-item]


def _alpaca(path: Path, n: int = 3) -> Path:
    rows = [{"instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_cli_meta(tmp_path: Path, capsys) -> None:
    src, out = _alpaca(tmp_path / "in.jsonl"), tmp_path / "o.jsonl"
    main(["convert", "-i", str(src), "-o", str(out), "--from", "alpaca", "--format",
          "messages", "--meta", "dataset=kullm", "--meta", "license=apache-2.0"])  # fmt: skip
    row = json.loads(out.read_text().splitlines()[0])
    assert row["meta"] == {"dataset": "kullm", "license": "apache-2.0"}
    with pytest.raises(SystemExit) as e:
        main(["convert", "-i", str(src), "-o", str(out), "--from", "alpaca", "--format",
              "messages", "--meta", "novalue"])  # fmt: skip
    assert e.value.code == 2


def test_recipe_license_report(tmp_path: Path) -> None:
    _alpaca(tmp_path / "a.jsonl", 3)
    _alpaca(tmp_path / "b.jsonl", 2)
    recipe = parse_recipe(
        {
            "output": "out.jsonl",
            "sources": {
                "a": {"path": "a.jsonl", "license": "apache-2.0",
                      "convert": {"from": "alpaca", "meta": {"dataset": "a"}}},
                "b": {"path": "b.jsonl", "license": "cc-by-nc-4.0",
                      "convert": {"from": "alpaca"}},
            },
            "mix": {"weights": {"a": 1, "b": 1}},
        },
        path=tmp_path / "recipe.yaml",
    )  # fmt: skip
    logs: list[str] = []
    result = run(recipe, log=logs.append)
    assert result.report["licenses"] == {
        "a": {"license": "apache-2.0", "source": "declared", "rows": 3},
        "b": {"license": "cc-by-nc-4.0", "source": "declared", "rows": 2,
              "warning": "non-commercial or research-only terms"},
    }  # fmt: skip
    assert any(line.startswith("[license] b (cc-by-nc-4.0)") for line in logs)
    rows = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert sum(r.get("meta") == {"dataset": "a"} for r in rows) == 3
    # Skipped (up-to-date) steps keep their license information.
    again = run(recipe, log=lambda _m: None)
    assert again.ran == [] and again.report["licenses"] == result.report["licenses"]

    with pytest.raises(RecipeError, match="meta"):
        parse_recipe(
            {"output": "o.jsonl", "sources": {"a": {"path": "a.jsonl",
             "convert": {"from": "alpaca", "meta": {"x": [1]}}}}},
            path=tmp_path / "r.yaml",
        )  # fmt: skip


def test_recipe_detects_hf_license(tmp_path: Path, monkeypatch) -> None:
    import convmerge.fetch.hf as hf
    import convmerge.licenses as licenses

    def fake_download(dataset_id, dst, **kwargs):
        return _alpaca(Path(dst), 2)

    monkeypatch.setattr(hf, "download_hf_dataset", fake_download)
    monkeypatch.setattr(licenses, "detect_hf_license", lambda repo, token=None: "mit")
    recipe = parse_recipe(
        {"output": "o.jsonl",
         "sources": {"kullm": {"fetch": {"hf": "nlpai-lab/kullm-v2", "max_rows": 2},
                               "convert": {"from": "alpaca"}}}},
        path=tmp_path / "r.yaml",
    )  # fmt: skip
    report = run(recipe, log=lambda _m: None).report
    assert report["licenses"] == {"kullm": {"license": "mit", "source": "card", "rows": 2}}
    assert report["license_warnings"] == []
