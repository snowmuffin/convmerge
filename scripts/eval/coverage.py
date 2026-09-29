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

# Picked for the 1.1 evaluation from the Hub's trending and most-downloaded
# lists (September 2026): never looked at while building convmerge.
FRESH: list[tuple[str, str, str | None]] = [
    # general / agent SFT
    ("nvidia/Nemotron-SFT-Instruction-Following-Chat-v3", "sft", None),
    ("allenai/tulu-3-sft-personas-instruction-following", "sft", None),
    ("openbmb/UltraData-SFT-Agent-2609", "sft", "Tool-Use"),
    ("OpenDataArena/Spark-234K", "sft", None),
    ("nisten/opus5-5-doctor-patient-conversations-all-human-diseases", "sft", None),
    ("OpenAssistant/oasst2", "sft", None),
    ("CohereLabs/aya_dataset", "sft", None),
    ("HuggingFaceTB/smol-smoltalk", "sft", None),
    ("agent-eto/eto-sft-trajectory", "sft", None),
    # reasoning
    ("MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", "reasoning", None),
    ("IFM/Code-Reasoning", "reasoning", "code-thinking-v1"),
    ("CohereLabs/tiny-aya-l2-thinker-multilingual-reasoning", "reasoning", "ko"),
    ("Roman1111111/GPT-5.6-luna-reasoning-102881x", "reasoning", None),
    ("open-thoughts/OpenThoughts-114k", "reasoning", None),
    # tools
    ("zake7749/Qwen3.6-35B-A3B-Tool-Calling", "tools", None),
    ("zake7749/deepseek-v4-pro-agent-tool-calling-trajectory", "tools", None),
    ("smolagents/toolcalling", "tools", None),
    ("Mozilla/standard_chat_tool_calling_general", "tools", None),
    ("younissk/tool-calling-mix", "tools", None),
    ("ZeroAgency/gemma3-pythonic-function-tool-calling-v1", "tools", None),
    # preference
    ("argilla/distilabel-math-preference-dpo", "preference", None),
    ("shibing624/DPO-En-Zh-20k-Preference", "preference", "en"),
    ("Columbia-NLP/DPO-tldr-summarisation-preferences", "preference", None),
    ("allenai/llama-3.1-tulu-3-8b-preference-mixture", "preference", None),
    ("openbmb/UltraFeedback", "preference", None),
    # Korean
    ("CertifiedJoon/Korean-Instruction", "sft", None),
    ("neuralfoundry-coder/korean-legal-instruction-sample", "sft", None),
    ("heegyu/open-korean-instructions-v20231020", "sft", None),
    ("ChuGyouk/argilla-distilabel-math-preference-dpo-korean", "preference", None),
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


RAW_ROWS = 0  # --raw: source rows kept in the result (strings clipped)


def _clip(value: Any, limit: int = 300) -> Any:
    if isinstance(value, str) and value[:1] in "[{":
        try:  # a JSON column (messages_json): clip inside it, not the whole string
            return {"<json>": _clip(json.loads(value), limit)}
        except ValueError:
            pass
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"...(+{len(value) - limit})"
    if isinstance(value, dict):
        return {k: _clip(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit) for v in value[:8]]
    return value if isinstance(value, (int, float, bool, type(None))) else str(value)


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
    result["raw"] = [_clip(r) for r in data[:RAW_ROWS]]
    ok = stats.lines_read and stats.written >= 0.9 * stats.lines_read
    result["status"] = "ok" if ok else ("partial" if stats.written else "fail")
    return result


def _isolated(dataset: str, kind: str, config: str | None, args: argparse.Namespace) -> dict:
    """``check`` in a child process with a time and resident-memory cap.

    A dataset whose loader hangs or exhausts memory (one took the whole runner
    down) is recorded as a load error instead of ending the run.
    """
    import subprocess
    import time

    cmd = [sys.executable, __file__, "--rows", str(args.rows), "--raw", str(args.raw),
           "--one", dataset, kind, config or "-"]  # fmt: skip
    base: dict[str, Any] = {"id": dataset, "kind": kind, "config": config}
    limit_kb = int(args.max_memory_gb * 2**20)
    deadline = time.monotonic() + args.timeout
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        reason = None
        while proc.poll() is None:
            if time.monotonic() > deadline:
                reason = f"timed out after {args.timeout}s"
            elif _rss_kb(proc.pid) > limit_kb:
                reason = f"used more than {args.max_memory_gb:g} GB of memory"
            if reason:
                proc.kill()
                proc.wait()
                return {**base, "status": "load_error", "note": reason}
            time.sleep(0.5)
        out, err = proc.communicate()
    lines = out.strip().splitlines()
    if proc.returncode == 0 and lines:
        return json.loads(lines[-1])
    tail = err.strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
    return {**base, "status": "load_error", "note": f"child failed: {tail[0]}"[:300]}


def _rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _shape(value: Any) -> str:
    if isinstance(value, list):
        inner = _shape(value[0]) if value else "?"
        return f"list[{inner}]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_shape(v)}" for k, v in list(value.items())[:8]) + "}"
    return type(value).__name__


def render(results: list[dict[str, Any]], n: int) -> str:
    lines = [f"### Zero-config coverage: `convert --from auto`, first {n} rows", ""]
    for name in dict.fromkeys(r.get("set", "") for r in results):
        part = [r for r in results if r.get("set", "") == name]
        loadable = [r for r in part if r["status"] != "load_error"]
        ok = sum(r["status"] == "ok" for r in loadable)
        lines.append(f"- {name or 'all'}: ok {ok} / loadable {len(loadable)} "
                     f"(load errors: {len(part) - len(loadable)})")  # fmt: skip
    lines += [
        "",
        "| Set | Dataset | Kind | Status | Written | Drops / note | Columns |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        drops = ", ".join(f"{k}={v}" for k, v in sorted(r.get("drops", {}).items()))
        note = drops or r.get("note", "")
        written = f"{r.get('written', '')}/{r.get('read', '')}" if "read" in r else ""
        cols = ", ".join(r.get("columns", []))[:120]
        cells = [r.get("set", ""), r["id"], r["kind"], r["status"], written, note[:160], cols]
        cells = [c.replace("|", "\\|").replace("\n", " ") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--one", nargs=3, metavar=("DATASET", "KIND", "CONFIG"),
                        help=argparse.SUPPRESS)  # fmt: skip
    parser.add_argument("--timeout", type=int, default=300, help="seconds per dataset")
    parser.add_argument("--max-memory-gb", type=float, default=4.0)
    parser.add_argument("--only", nargs="*", default=None, metavar="DATASET",
                        help="check only these dataset ids")  # fmt: skip
    parser.add_argument("--raw", type=int, default=0, metavar="N",
                        help="print N source rows of datasets that fail")  # fmt: skip
    parser.add_argument("--set", choices=("v014", "fresh", "all"), default="all",
                        help="v014: the 0.14 list; fresh: the 1.1 additions")  # fmt: skip
    args = parser.parse_args()
    global RAW_ROWS
    RAW_ROWS = args.raw
    if args.one:
        dataset, kind, config = args.one
        print(json.dumps(check(dataset, kind, None if config == "-" else config, args.rows)))
        return 0
    sets = {"v014": DATASETS, "fresh": FRESH}
    results = []
    for name, entries in sets.items():
        if args.set not in (name, "all"):
            continue
        for dataset, kind, config in entries:
            if args.only and dataset not in args.only:
                continue
            print(f"[check] {dataset}", file=sys.stderr, flush=True)
            results.append({**_isolated(dataset, kind, config, args), "set": name})
    text = render(results, args.rows)
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as f:
            f.write(text)
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in results:  # full detail for failures, in the log
        if r["status"] not in ("ok",):
            raw = r.pop("raw", None)
            print(json.dumps(r, ensure_ascii=False)[:1500])
            for row in raw or []:
                print("  raw:", json.dumps(row, ensure_ascii=False)[:3000])
    return 0


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    sys.exit(main())
