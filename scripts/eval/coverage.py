"""Zero-config coverage: convert popular Hub datasets that are NOT in the catalog.

For each dataset, stream the first N rows, run ``convert --from auto`` with no
flags other than the output format the dataset kind implies, and record how
many rows convert and why the rest drop. Writes a Markdown table and a JSON
file with the raw column names of every dataset, so failures can be traced
to their layout. Evaluation harness, not part of the package.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

# (dataset id, kind, config or None)
DATASETS: list[tuple[str, str, str | None]] = [
    # general SFT
    ("mlabonne/FineTome-100k", "sft", None),
    ("HuggingFaceH4/no_robots", "sft", None),
    ("Open-Orca/OpenOrca", "sft", None),
    ("WizardLMTeam/WizardLM_evol_instruct_V2_196k", "sft", None),
    ("arcee-ai/The-Tome", "sft", None),
    ("microsoft/orca-agentinstruct-1M-v1", "sft", None),
    ("allenai/WildChat-1M", "sft", None),
    ("BAAI/Infinity-Instruct", "sft", "0625"),
    ("facebook/natural_reasoning", "sft", None),
    ("openbmb/UltraInteract_sft", "sft", None),
    # math / code
    ("openai/gsm8k", "sft", "main"),
    ("nvidia/OpenMathInstruct-2", "sft", None),
    ("AI-MO/NuminaMath-CoT", "sft", None),
    ("TIGER-Lab/MathInstruct", "sft", None),
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", "sft", None),
    ("m-a-p/CodeFeedback-Filtered-Instruction", "sft", None),
    ("glaiveai/glaive-code-assistant", "sft", None),
    # reasoning
    ("bespokelabs/Bespoke-Stratos-17k", "reasoning", None),
    ("NovaSky-AI/Sky-T1_data_17k", "reasoning", None),
    ("GAIR/LIMO", "reasoning", None),
    ("open-r1/Mixture-of-Thoughts", "reasoning", "math"),
    # tools
    ("Team-ACE/ToolACE", "tools", None),
    ("Locutusque/function-calling-chatml", "tools", None),
    # preference
    ("argilla/ultrafeedback-binarized-preferences-cleaned", "preference", None),
    ("mlabonne/orpo-dpo-mix-40k", "preference", None),
    ("jondurbin/truthy-dpo-v0.1", "preference", None),
    ("PKU-Alignment/PKU-SafeRLHF", "preference", None),
    ("stanfordnlp/SHP", "preference", None),
    ("nvidia/HelpSteer3", "preference", "preference"),
    ("Unified-Language-Model-Alignment/Anthropic_HH_Golden", "preference", None),
    # Korean
    ("squarelike/OpenOrca-gugugo-ko", "sft", None),
    ("nlpai-lab/openassistant-guanaco-ko", "sft", None),
    ("MarkrAI/KoCommercial-Dataset", "sft", None),
    ("kyujinpy/KOpen-platypus", "sft", None),
    ("maywell/ko_wikidata_QA", "sft", None),
    ("jojo0217/korean_rlhf_dataset", "sft", None),
]


def rows(dataset: str, config: str | None, n: int) -> tuple[str, list[dict[str, Any]]]:
    from datasets import get_dataset_split_names, load_dataset

    try:
        splits = get_dataset_split_names(dataset, config)
    except Exception:  # noqa: BLE001 - fall back to "train"
        splits = ["train"]
    split = "train" if "train" in splits else splits[0]
    ds = load_dataset(dataset, config, split=split, streaming=True)
    return split, [dict(r) for r in ds.take(n)]


def check(dataset: str, kind: str, config: str | None, n: int) -> dict[str, Any]:
    from convmerge import ConvertStats, build_convert_config, convert_with_config

    result: dict[str, Any] = {"id": dataset, "kind": kind, "config": config}
    try:
        split, data = rows(dataset, config, n)
    except Exception as exc:  # noqa: BLE001
        result.update(status="load_error", note=f"{type(exc).__name__}: {exc}"[:300])
        return result
    result["split"] = split
    result["columns"] = sorted({k for r in data for k in r})
    result["sample"] = {k: _shape(v) for k, v in (data[0] if data else {}).items()}
    fmt = "preference" if kind == "preference" else "messages"
    stats = ConvertStats()
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp, "in.jsonl"), Path(tmp, "out.jsonl")
        src.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n"
                               for r in data), encoding="utf-8")  # fmt: skip
        try:
            convert_with_config(src, dst, build_convert_config(adapter="auto", output_format=fmt),
                                stats=stats)  # fmt: skip
        except Exception:  # noqa: BLE001
            result.update(status="crash", note=traceback.format_exc()[-600:])
            return result
        first = dst.read_text(encoding="utf-8").splitlines()[:1]
    result.update(read=stats.lines_read, written=stats.written, reasoning=stats.reasoning,
                  drops=dict(stats.drop_reasons))  # fmt: skip
    result["first"] = first[0][:400] if first else None
    ok = stats.lines_read and stats.written >= 0.9 * stats.lines_read
    result["status"] = "ok" if ok else ("partial" if stats.written else "fail")
    return result


def _shape(value: Any) -> str:
    if isinstance(value, list):
        inner = _shape(value[0]) if value else "?"
        return f"list[{inner}]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_shape(v)}" for k, v in list(value.items())[:8]) + "}"
    return type(value).__name__


def render(results: list[dict[str, Any]], n: int) -> str:
    loadable = [r for r in results if r["status"] != "load_error"]
    ok = sum(r["status"] == "ok" for r in loadable)
    lines = [
        f"### Zero-config coverage: `convert --from auto`, first {n} rows",
        "",
        f"ok {ok} / loadable {len(loadable)} (load errors: {len(results) - len(loadable)})",
        "",
        "| Dataset | Kind | Status | Written | Drops / note | Columns |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        drops = ", ".join(f"{k}={v}" for k, v in sorted(r.get("drops", {}).items()))
        note = drops or r.get("note", "")
        written = f"{r.get('written', '')}/{r.get('read', '')}" if "read" in r else ""
        cols = ", ".join(r.get("columns", []))[:120]
        cells = [r["id"], r["kind"], r["status"], written, note[:160], cols]
        cells = [c.replace("|", "\\|").replace("\n", " ") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = []
    for dataset, kind, config in DATASETS:
        print(f"[check] {dataset}", file=sys.stderr, flush=True)
        results.append(check(dataset, kind, config, args.rows))
    text = render(results, args.rows)
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(text)
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in results:  # full detail for failures, in the log
        if r["status"] not in ("ok",):
            print(json.dumps(r, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    sys.exit(main())
