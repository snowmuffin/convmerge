"""The tested-datasets catalog (``tests/datasets/catalog.json``).

``python scripts/datasets.py table [--write]``
    Print the README "Tested datasets" table, or rewrite it in README.md.

``python scripts/datasets.py check [--rows N] [--only ID ...] [--summary PATH] [--show-drops N]``
    Stream the first N rows of every catalog dataset from the Hugging Face
    Hub (needs ``pip install "convmerge[fetch-all]"`` and network access),
    convert them the way the catalog says, and report how many converted and
    why the rest were dropped. Gated datasets are skipped unless ``HF_TOKEN``
    is set. A dataset fails when it cannot be loaded, when fewer than
    ``--min-ok`` of its rows convert, or, for reasoning datasets, when no
    converted row carries a reasoning trace. ``--summary`` appends a Markdown
    table (the ``datasets`` workflow passes ``$GITHUB_STEP_SUMMARY``). Exits 1
    if any dataset fails. ``--show-drops N`` prints up to N dropped rows per
    dataset (long strings shortened) to find out why they drop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "tests" / "datasets" / "catalog.json"
README = ROOT / "README.md"
START, END = "<!-- datasets:start -->", "<!-- datasets:end -->"
KINDS = {
    "sft": "SFT",
    "reasoning": "Reasoning",
    "tools": "Tool calling",
    "preference": "Preference",
}


def load_catalog() -> list[dict[str, Any]]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def sample_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """The entry's made-up rows: ``records`` when one example spans several rows
    (OpenAssistant message tables), else ``[record]``."""
    return list(entry.get("records") or [entry["record"]])


def flags(entry: dict[str, Any]) -> str:
    out = f"--from {entry.get('adapter', 'auto')} --format {entry['format']}"
    if entry.get("preference"):
        out += f" --preference {entry['preference']}"
    for key, value in {**entry.get("emit", {}), **entry.get("transforms", {})}.items():
        out += f" --{key.replace('_', '-')} {value}"
    if entry.get("adapter_kwargs"):
        out += f" --adapter-kwargs '{json.dumps(entry['adapter_kwargs'], separators=(',', ':'))}'"
    return out


def convert_config(entry: dict[str, Any]) -> Any:
    """The ``ConvertConfig`` the catalog entry's flags describe."""
    from convmerge import build_convert_config

    kwargs = entry.get("adapter_kwargs")
    return build_convert_config(
        adapter=entry.get("adapter", "auto"),
        output_format=entry["format"],
        preference=entry.get("preference"),
        emit_overrides=entry.get("emit") or None,
        adapter_kwargs_json=json.dumps(kwargs) if kwargs else None,
        transform_overrides=entry.get("transforms") or None,
    )


def render_table(catalog: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Kind | Lang | Layout | `convmerge convert` flags |",
        "|---------|------|------|--------|---------------------------|",
    ]
    for kind, label in KINDS.items():
        for e in (e for e in catalog if e["kind"] == kind):
            name = f"[{e['id']}](https://huggingface.co/datasets/{e['id']})"
            if e.get("gated"):
                name += " (gated)"
            lang = e.get("lang", "en")
            lines.append(f"| {name} | {label} | {lang} | {e['shape']} | `{flags(e)}` |")
    return "\n".join(lines)


def write_readme(table: str) -> bool:
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"README.md has no {START} ... {END} block")
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    new = f"{head}{START}\n{table}\n{END}{tail}"
    README.write_text(new, encoding="utf-8")
    return new != text


@dataclass
class Result:
    id: str
    status: str  # ok | warn | fail | skip
    read: int = 0
    written: int = 0
    reasoning: int = 0
    drops: dict[str, int] = field(default_factory=dict)
    note: str = ""


Loader = Callable[[dict[str, Any], int], Iterable[dict[str, Any]]]


def hub_rows(entry: dict[str, Any], rows: int) -> Iterable[dict[str, Any]]:
    """The first ``rows`` rows of a catalog dataset, streamed from the Hub."""
    from datasets import load_dataset

    ds = load_dataset(entry["id"], entry.get("config"), split=entry.get("split") or "train",
                      streaming=True)  # fmt: skip
    return ds.take(rows)


def check_entry(
    entry: dict[str, Any],
    rows: int,
    *,
    loader: Loader = hub_rows,
    min_ok: float = 0.9,
    show_drops: int = 0,
) -> Result:
    """Convert real rows of one catalog dataset and judge the outcome."""
    from convmerge import ConvertStats, convert_with_config

    rid = entry["id"]
    if entry.get("live") is False:
        return Result(rid, "skip", note=entry.get("note", "not loadable with datasets"))
    if entry.get("gated") and not os.environ.get("HF_TOKEN"):
        return Result(rid, "skip", note="gated: set HF_TOKEN to check it")
    stats = ConvertStats()
    traced = 0  # raw rows that look like they carry a reasoning trace
    kept: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "in.jsonl"), Path(tmp, "out.jsonl")
            with src.open("w", encoding="utf-8") as f:
                for row in loader(entry, rows):
                    traced += has_trace(row)
                    if show_drops:
                        kept.append(row)
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            convert_with_config(src, dst, convert_config(entry), stats=stats)
    except Exception as exc:  # noqa: BLE001 - report it and go on with the next dataset
        return Result(rid, "fail", note=f"{type(exc).__name__}: {exc}"[:300])
    if show_drops:
        _print_drops(rid, kept, stats, show_drops)
    drops = dict(stats.drop_reasons)
    if stats.skipped:
        drops["skipped_lines"] = stats.skipped
    records = stats.lines_read - stats.grouped  # message rows of one tree make one record
    result = Result(rid, "ok", records, stats.written, stats.reasoning, drops)
    if stats.grouped:
        result.note = f"{stats.lines_read:,} message rows read as {records:,} trees"
    if not stats.lines_read:
        result.status, result.note = "fail", "no rows"
    elif stats.written < entry.get("min_ok", min_ok) * records:
        need = entry.get("min_ok", min_ok)
        result.status, result.note = "fail", f"fewer than {need:.0%} of rows converted"
    elif entry["kind"] == "reasoning" and not stats.reasoning and traced:
        result.status = "fail"
        result.note = f"{traced} rows carry a reasoning trace, but no converted row does"
    elif entry["kind"] == "reasoning" and not stats.reasoning:
        result.status, result.note = "warn", "these rows carry no reasoning trace"
    elif stats.written < records:
        result.status = "warn"
    return result


def _print_drops(rid: str, rows: list[dict[str, Any]], stats: Any, limit: int) -> None:
    shown = 0
    for reason, lines in sorted(stats.drop_lines.items()):
        for number in lines:
            if shown >= limit or not 0 < number <= len(rows):
                return
            shown += 1
            print(f"--- {rid} line {number}: {reason}")
            print(json.dumps(_shorten(rows[number - 1]), ensure_ascii=False, default=str))


def _shorten(value: Any, width: int = 400) -> Any:
    if isinstance(value, str):
        return value if len(value) <= width else value[:width] + f"…(+{len(value) - width})"
    if isinstance(value, list):
        return [_shorten(v, width) for v in value]
    if isinstance(value, dict):
        return {k: _shorten(v, width) for k, v in value.items()}
    return value


_TRACE_KEY_PARTS = ("think", "reason", "thought", "trajectory")


def has_trace(value: Any, depth: int = 0) -> bool:
    """Whether a raw row looks like it holds a reasoning trace anywhere: a
    ``<think>`` block, or a long string under a key such as ``thinking``,
    ``reasoning_content``, or ``deepseek_thinking_trajectory`` (short values
    like Llama-Nemotron's ``reasoning: "on"`` flag do not count)."""
    if depth > 4:
        return False
    if isinstance(value, str):
        return "<think>" in value
    if isinstance(value, list):
        return any(has_trace(v, depth + 1) for v in value)
    if isinstance(value, dict):
        for k, v in value.items():
            named = any(part in str(k).lower() for part in _TRACE_KEY_PARTS)
            if named and isinstance(v, str) and len(v) >= 20:
                return True
            if has_trace(v, depth + 1):
                return True
    return False


def render_summary(results: list[Result], rows: int) -> str:
    icon = {"ok": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}
    counts = {k: sum(r.status == k for r in results) for k in icon}
    lines = [
        f"### Catalog check: first {rows} rows of each dataset",
        "",
        " · ".join(f"{icon[k]} {k} {n}" for k, n in counts.items() if n),
        "",
        "| | Dataset | Converted | Reasoning | Dropped / note |",
        "|---|---------|-----------|-----------|----------------|",
    ]
    for r in results:
        drops = ", ".join(f"{k}={n}" for k, n in sorted(r.drops.items()))
        detail = "; ".join(x for x in (drops, r.note.replace("|", "\\|")) if x)
        converted = f"{r.written}/{r.read}" if r.status != "skip" else ""
        reasoning = str(r.reasoning) if r.reasoning else ""
        lines.append(f"| {icon[r.status]} | {r.id} | {converted} | {reasoning} | {detail} |")
    return "\n".join(lines) + "\n"


def check(
    rows: int,
    only: list[str] | None,
    *,
    summary: Path | None = None,
    min_ok: float = 0.9,
    loader: Loader = hub_rows,
    show_drops: int = 0,
) -> int:
    results: list[Result] = []
    for e in load_catalog():
        if only and e["id"] not in only:
            continue
        r = check_entry(e, rows, loader=loader, min_ok=min_ok, show_drops=show_drops)
        results.append(r)
        detail = f"  dropped={r.drops}" if r.drops else ""
        note = f"  ({r.note})" if r.note else ""
        print(f"{r.status:<5} {r.id}: {r.written}/{r.read}{detail}{note}", flush=True)
    if summary is not None:
        with summary.open("a", encoding="utf-8") as f:
            f.write(render_summary(results, rows))
    return 1 if any(r.status == "fail" for r in results) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tested-datasets catalog tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    table = sub.add_parser("table", help="Print (or --write) the README dataset table")
    table.add_argument("--write", action="store_true", help="Rewrite the table in README.md")
    live = sub.add_parser("check", help="Convert real rows of every catalog dataset")
    live.add_argument("--rows", type=int, default=50)
    live.add_argument("--only", nargs="+", default=None, metavar="ID")
    live.add_argument("--min-ok", type=float, default=0.9, help="Share of rows that must convert")
    live.add_argument("--summary", type=Path, default=None, help="Append a Markdown table here")
    live.add_argument("--show-drops", type=int, default=0, metavar="N",
                      help="Print up to N dropped rows per dataset")  # fmt: skip
    args = parser.parse_args(argv)
    if args.cmd == "table":
        rendered = render_table(load_catalog())
        if args.write:
            print("README.md updated" if write_readme(rendered) else "README.md already up to date")
        else:
            print(rendered)
        return 0
    return check(args.rows, args.only, summary=args.summary, min_ok=args.min_ok,
                 show_drops=args.show_drops)  # fmt: skip


if __name__ == "__main__":
    # Run as a script, this file's folder is sys.path[0], and ``import datasets``
    # would find this file instead of the Hugging Face library.
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    sys.exit(main())
