"""Run the quality commands on real rows of the tested-datasets catalog.

``python scripts/quality.py [--rows N] [--only ID ...] [--summary PATH] [--json PATH]``

Streams the first N rows of each catalog dataset from the Hugging Face Hub,
converts them the way the catalog says, and reports what ``filter`` (every
rule, with Korean datasets also checked for ``script hangul=0.3``),
``decontam`` (against a few public benchmarks, all turns), and
``dedupe --near`` find, with example rows per rule so thresholds can be
judged by eye. It never fails on what it finds: it exists to measure false
positives. Needs ``pip install "convmerge[fetch-all,quality]"``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_SETS = (
    "hf:openai/gsm8k:main",
    "hf:cais/mmlu:all",
    "hf:HAERAE-HUB/KMMLU:Law",
    "hf:HAERAE-HUB/KMMLU:Korean-History",
    "hf:skt/kobest_v1:boolq",
)
OPTIONAL = ("slop",)


def _catalog_module() -> Any:
    # scripts/datasets.py shares its name with the Hugging Face library.
    spec = importlib.util.spec_from_file_location("_catalog", ROOT / "scripts" / "datasets.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_catalog"] = module
    spec.loader.exec_module(module)
    return module


def check(entry: dict[str, Any], rows: int, index: Any, catalog: Any) -> dict[str, Any]:
    from convmerge import (
        FilterSpec,
        FilterStats,
        NearDedupeStats,
        convert_with_config,
        decontaminate_jsonl,
        deduplicate_near_jsonl,
        filter_jsonl,
    )
    from convmerge.quality import DEFAULT_RULES

    result: dict[str, Any] = {"id": entry["id"], "lang": entry.get("lang", "en")}
    with tempfile.TemporaryDirectory() as tmp:
        raw, conv = Path(tmp, "raw.jsonl"), Path(tmp, "conv.jsonl")
        with raw.open("w", encoding="utf-8") as f:
            for row in catalog.hub_rows(entry, rows):
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        convert_with_config(raw, conv, catalog.convert_config(entry))
        korean = "ko" in str(entry.get("lang", ""))
        spec = FilterSpec.from_options(enable=OPTIONAL,
                                       min_script={"hangul": 0.3} if korean else None)  # fmt: skip
        st = FilterStats()
        filter_jsonl(conv, spec=spec, stats=st)
        report = st.to_report()
        result["rows"] = st.rows
        result["rules"] = report["rules"]
        result["samples"] = report["samples"]
        result["dropped_by_default"] = _dropped(conv, DEFAULT_RULES)
        if "preference" in report:
            result["preference"] = report["preference"]
        dec = decontaminate_jsonl(conv, index, check="all")
        result["decontam"] = {k: v["matched"] for k, v in dec.eval_sets.items() if v["matched"]}
        result["decontam_samples"] = dec.samples
        near = NearDedupeStats()
        deduplicate_near_jsonl(conv, Path(tmp, "near.jsonl"), stats=near)
        result["near_duplicates"] = near.near_duplicates
    return result


def _dropped(path: Path, rules: tuple[str, ...]) -> int:
    from convmerge import FilterSpec, filter_jsonl

    return filter_jsonl(path, spec=FilterSpec(rules=rules)).rejected


def render(results: list[dict[str, Any]], rows: int) -> str:
    rules = ["empty_answer", "refusal", "repetition", "near_identical_pair", "rejected_empty",
             "slop", "script"]  # fmt: skip
    head = ["Dataset", "Rows", "Dropped (defaults)", *rules, "Decontam", "Near-dup"]
    lines = [
        f"### Quality check: first {rows} rows of each dataset",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['id']} | error: {r['error'][:120]} |" + " |" * (len(head) - 2))
            continue
        hits = [str(r["rules"].get(rule, "")) or "" for rule in rules]
        decontam = ", ".join(f"{k.split(':')[1]}={v}" for k, v in r["decontam"].items())
        cells = [r["id"], str(r["rows"]), str(r["dropped_by_default"]), *hits, decontam,
                 str(r["near_duplicates"])]  # fmt: skip
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    lines += ["", "<details><summary>Examples per rule</summary>", ""]
    for rule in [*rules, "decontam"]:
        examples = []
        for r in results:
            if rule == "decontam":
                examples += [(r["id"], s["line"], f"[{s['eval']}] {s['text']}")
                             for s in r.get("decontam_samples", [])]  # fmt: skip
            else:
                examples += [(r["id"], s["line"], s["text"])
                             for s in r.get("samples", {}).get(rule, [])]  # fmt: skip
        if not examples:
            continue
        lines += [f"**{rule}**", ""]
        for rid, line, text in examples[:30]:
            text = text.replace("\n", " ").replace("|", "\\|")[:200]
            lines.append(f"- `{rid}` line {line}: {text}")
        lines.append("")
    lines.append("</details>")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--only", nargs="+", default=None, metavar="ID")
    parser.add_argument("--summary", type=Path, default=None, help="Append Markdown here")
    parser.add_argument("--json", type=Path, default=None, help="Write full results here")
    args = parser.parse_args(argv)

    from convmerge import EvalSource, build_index

    catalog = _catalog_module()
    token = os.environ.get("HF_TOKEN") or None
    index = build_index([EvalSource(s) for s in EVAL_SETS], token=token)
    for name, counts in zip(index.names, index.counts):
        print(f"[eval] {name}: {counts}", file=sys.stderr)
    results = []
    for entry in catalog.load_catalog():
        if args.only and entry["id"] not in args.only:
            continue
        if entry.get("live") is False or (entry.get("gated") and not token):
            continue
        print(f"[check] {entry['id']}", file=sys.stderr)
        try:
            results.append(check(entry, args.rows, index, catalog))
        except Exception as exc:  # noqa: BLE001 - report it and go on
            results.append({"id": entry["id"], "error": f"{type(exc).__name__}: {exc}"})
    text = render(results, args.rows)
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(text)
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    sys.exit(main())
