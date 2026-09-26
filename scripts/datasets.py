"""The tested-datasets catalog (``tests/datasets/catalog.json``).

``python scripts/datasets.py table [--write]``
    Print the README "Tested datasets" table, or rewrite it in README.md.

``python scripts/datasets.py check [--rows N] [--only ID ...]``
    Stream the first N rows of every catalog dataset from the Hugging Face
    Hub (needs ``pip install "convmerge[fetch-all]"``, network access, and
    ``HF_TOKEN`` for gated ones), convert them the way the catalog says, and
    report how many converted and why the rest were dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "tests" / "datasets" / "catalog.json"
README = ROOT / "README.md"
START, END = "<!-- datasets:start -->", "<!-- datasets:end -->"
KINDS = {"sft": "SFT", "tools": "Tool calling", "preference": "Preference"}


def load_catalog() -> list[dict[str, Any]]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def flags(entry: dict[str, Any]) -> str:
    out = f"--from auto --format {entry['format']}"
    if entry.get("preference"):
        out += f" --preference {entry['preference']}"
    return out


def render_table(catalog: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Kind | Layout | `convmerge convert` flags |",
        "|---------|------|--------|---------------------------|",
    ]
    for kind, label in KINDS.items():
        for e in (e for e in catalog if e["kind"] == kind):
            name = f"[{e['id']}](https://huggingface.co/datasets/{e['id']})"
            if e.get("gated"):
                name += " (gated)"
            lines.append(f"| {name} | {label} | {e['shape']} | `{flags(e)}` |")
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


def check(rows: int, only: list[str] | None) -> int:
    from datasets import load_dataset

    from convmerge import ConvertStats, build_convert_config, convert_with_config

    failures = 0
    for e in load_catalog():
        if only and e["id"] not in only:
            continue
        if e.get("live") is False:
            print(f"skip  {e['id']}: {e.get('note', 'not loadable with datasets')}")
            continue
        try:
            ds = load_dataset(e["id"], e.get("config"), split=e.get("split") or "train",
                              streaming=True)  # fmt: skip
            with tempfile.TemporaryDirectory() as tmp:
                src, dst = Path(tmp, "in.jsonl"), Path(tmp, "out.jsonl")
                with src.open("w", encoding="utf-8") as f:
                    for row in ds.take(rows):
                        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                cfg = build_convert_config(
                    adapter="auto", output_format=e["format"], preference=e.get("preference")
                )
                stats = ConvertStats()
                convert_with_config(src, dst, cfg, stats=stats)
        except Exception as exc:  # noqa: BLE001 - report it and go on with the next dataset
            failures += 1
            print(f"ERROR {e['id']}: {exc}")
            continue
        ok = stats.written == stats.lines_read
        failures += not ok
        detail = "" if ok else f"  dropped={stats.drop_reasons} skipped={stats.skipped}"
        print(f"{'ok   ' if ok else 'FAIL '} {e['id']}: {stats.written}/{stats.lines_read}{detail}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tested-datasets catalog tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    table = sub.add_parser("table", help="Print (or --write) the README dataset table")
    table.add_argument("--write", action="store_true", help="Rewrite the table in README.md")
    live = sub.add_parser("check", help="Convert real rows of every catalog dataset")
    live.add_argument("--rows", type=int, default=50)
    live.add_argument("--only", nargs="+", default=None, metavar="ID")
    args = parser.parse_args(argv)
    if args.cmd == "table":
        rendered = render_table(load_catalog())
        if args.write:
            print("README.md updated" if write_readme(rendered) else "README.md already up to date")
        else:
            print(rendered)
        return 0
    return check(args.rows, args.only)


if __name__ == "__main__":
    sys.exit(main())
