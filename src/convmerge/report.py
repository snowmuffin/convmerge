"""Readable build reports and dataset cards from what ``convmerge run`` recorded.

:func:`summarize_run` reads a recipe's JSON report (and, unless told not to,
its output file for row shapes) into one summary: where each source came
from and under which license, how many rows each step took in and let
through and why it dropped the rest, with examples, and the turn and answer
length distributions of the result. :func:`render_markdown` and
:func:`render_html` (one self-contained page, no scripts) show it;
:func:`dataset_card` writes a dataset card (``README.md`` with Hub front
matter) naming the sources, their licenses, and how to rebuild the file.
Nothing is uploaded.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

__all__ = ["dataset_card", "render_html", "render_markdown", "summarize_run"]

_TURN_BUCKETS = ("1", "2", "3", "4", "5+")
_LENGTH_BUCKETS = ((50, "< 50"), (200, "50–199"), (1000, "200–999"), (5000, "1k–5k"), (None, "5k+"))
_SIZES = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
    (100_000_000, "10M<n<100M"),
    (1_000_000_000, "100M<n<1B"),
)


def summarize_run(recipe: Any, *, scan: bool = True) -> dict[str, Any]:
    """Summarize the last ``convmerge run`` of ``recipe`` (a loaded :class:`Recipe`).

    Reads ``recipe.report_path``; raises ``FileNotFoundError`` when the recipe
    has not been run. ``scan=False`` skips reading the output file, so the
    summary has no ``shape``.
    """
    report = json.loads(Path(recipe.report_path).read_text(encoding="utf-8"))
    steps: dict[str, Any] = report.get("steps", {})
    licenses: dict[str, Any] = report.get("licenses", {})
    mixed = steps.get("mix", {}).get("stats", {}).get("sources", {})
    sources = []
    for name, src in recipe.sources.items():
        conv = steps.get(f"{name}.convert", {}).get("stats", {})
        lic = licenses.get(name, {})
        sources.append(
            {
                "name": name,
                "origin": _origin(src, recipe.base_dir),
                "license": lic.get("license"),
                "license_source": lic.get("source"),
                "warning": lic.get("warning"),
                "read": conv.get("lines_read"),
                "converted": conv.get("written"),
                "mixed": mixed.get(name, {}).get("written"),
            }
        )
    output = dict(report.get("output", {}))
    summary: dict[str, Any] = {
        "convmerge": report.get("convmerge"),
        "recipe": _display(Path(recipe.path), recipe.base_dir),
        "lock": _display(Path(recipe.lock_path), recipe.base_dir),
        "output": output,
        "sources": sources,
        "steps": [s for name, entry in steps.items() if (s := _step(name, entry)) is not None],
        "license_warnings": list(report.get("license_warnings", [])),
        "shape": None,
    }
    if scan and Path(recipe.output).is_file():
        summary["shape"] = _shape(Path(recipe.output))
    return summary


def _origin(src: Any, base: Path) -> str:
    entry = getattr(src, "fetch", None)
    if entry is not None and entry.hf:
        parts = [f"hf:{entry.hf}"]
        parts += [
            f"{k} {v}" for k in ("config", "split", "revision") if (v := getattr(entry, k, None))
        ]
        return ", ".join(parts)
    if entry is not None and entry.url:
        return str(entry.url)
    return _display(Path(src.path), base) if getattr(src, "path", None) else "?"


def _display(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return str(path)


def _step(name: str, entry: dict[str, Any]) -> dict[str, Any] | None:
    """One row of the pipeline table: rows in, rows out, why the rest went."""
    st = entry.get("stats") or {}
    kind = name.rsplit(".", 1)[-1] if "." in name else name
    reasons: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    note = None
    if kind == "convert":
        rows_in, rows_out = st.get("lines_read"), st.get("written")
        reasons = dict(st.get("drop_reasons", {}))
        reasons.update(_nonzero(st, "invalid_json", "non_object", "no_example"))
        for reason, lines in st.get("drop_lines", {}).items():
            samples += [{"reason": reason, "line": n, "text": ""} for n in lines[:3]]
    elif name == "mix":
        rows_in = sum(s.get("available", 0) for s in st.get("sources", {}).values())
        rows_out = st.get("total_written")
        if rows_in and rows_out is not None and rows_out < rows_in:
            reasons = {"not sampled": rows_in - rows_out}
        elif rows_in and rows_out is not None and rows_out > rows_in:
            note = "oversampled"
    elif name == "dedupe":
        rows_in, rows_out = st.get("total"), st.get("kept")
        reasons = _nonzero(st, "duplicates", "near_duplicates", "invalid_json")
    elif name == "filter":
        rows_in, rows_out = st.get("rows"), st.get("kept")
        reasons = {
            **_nonzero(st.get("rules", {}), *st.get("rules", {})),
            **_nonzero(st, "unreadable", "invalid_json"),
        }
        for rule, items in st.get("samples", {}).items():
            samples += [{"reason": rule, **s} for s in items]
        if len(st.get("rules", {})) and sum(1 for v in st["rules"].values() if v) > 1:
            note = "a row can match several rules"
    elif name == "decontam":
        rows_in, rows_out = st.get("rows"), st.get("kept")
        reasons = _nonzero(st, "contaminated", "unreadable", "invalid_json")
        samples = [
            {
                "reason": f"contaminated ({s.get('eval')})",
                "line": s.get("line"),
                "text": s.get("text", ""),
            }
            for s in st.get("samples", [])
        ]
    elif name == "tokens":
        rows_in, rows_out = st.get("rows"), st.get("kept")
        errors = sum(st.get("template_errors", {}).values())
        reasons = {
            **_nonzero(st, "over_limit", "unreadable", "invalid_json"),
            **({"template_errors": errors} if errors else {}),
        }
    elif name == "order":
        rows_in = rows_out = st.get("rows")
        reasons = {}
        note = f"by {st.get('by')}" + (f", {len(st['stages'])} stages" if st.get("stages") else "")
    elif name == "render":
        rows_in, rows_out = st.get("rows"), st.get("written")
        errors = sum(st.get("template_errors", {}).values())
        reasons = {
            **_nonzero(
                st, "pairs", "unstable_spans", "no_trained_turn", "unreadable", "invalid_json"
            ),
            **({"template_errors": errors} if errors else {}),
        }
    elif name.startswith("split."):
        part = name.split(".", 1)[1]
        rows_in, rows_out = st.get("total"), st.get(part)
        note = f"{part} part"
    else:  # fetch, normalize, output: no rows dropped
        return None
    removed = None
    if rows_in is not None and rows_out is not None and not name.startswith("split."):
        removed = max(rows_in - rows_out, 0)
    return {
        "name": name,
        "action": entry.get("action"),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "removed": removed,
        "reasons": reasons,
        "note": note,
        "samples": samples,
    }


def _nonzero(st: dict[str, Any], *keys: str) -> dict[str, int]:
    return {k: st[k] for k in keys if isinstance(st.get(k), int) and st[k]}


def _shape(path: Path) -> dict[str, Any]:
    """Turns per row and final-answer length, read from the output file."""
    from convmerge.adapter_resolve import resolve_adapter

    adapter = resolve_adapter("auto", None, pairs=True)
    turns = dict.fromkeys(_TURN_BUCKETS, 0)
    lengths: list[int] = []
    rows = pairs = unreadable = 0
    layout = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                unreadable += 1
                continue
            if layout is None and isinstance(value, dict):
                layout = _layout(value)
            examples = (
                [ex for ex in adapter(value) if ex.messages] if isinstance(value, dict) else []
            )
            if not examples:
                unreadable += 1
                continue
            ex = examples[0]
            if isinstance(value, dict) and "chosen" in value and "rejected" in value:
                pairs += 1
            users = sum(1 for m in ex.messages if m.role == "user")
            turns[_TURN_BUCKETS[min(max(users, 1), 5) - 1]] += 1
            answer = next((m for m in reversed(ex.messages) if m.role == "assistant"), None)
            lengths.append(len(_text(answer.content)) if answer is not None else 0)
    lengths.sort()
    hist = {label: 0 for _, label in _LENGTH_BUCKETS}
    for n in lengths:
        hist[next(label for top, label in _LENGTH_BUCKETS if top is None or n < top)] += 1
    pick = lambda p: lengths[min(len(lengths) - 1, int(p * len(lengths)))] if lengths else None  # noqa: E731
    return {
        "rows": rows,
        "layout": layout,
        "pairs": pairs,
        "unreadable": unreadable,
        "turns": turns,
        "answer_chars": {
            "p10": pick(0.1),
            "p50": pick(0.5),
            "p90": pick(0.9),
            "max": lengths[-1] if lengths else None,
        },
        "answer_histogram": hist,
    }


def _layout(row: dict[str, Any]) -> str:
    if "chosen" in row and "rejected" in row:
        return "preference"
    if "messages" in row:
        return "messages"
    if "conversations" in row:
        return "sharegpt"
    if "instruction" in row:
        return "alpaca"
    return "other"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(getattr(p, "text", None) or "" for p in content)
    return ""


# --- Markdown -------------------------------------------------------------------


def _n(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "–" if value is None else str(value)


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _why(step: dict[str, Any]) -> str:
    parts = [f"{k} {_n(v)}" for k, v in sorted(step["reasons"].items(), key=lambda kv: -kv[1])]
    if step.get("note"):
        parts.append(f"({step['note']})")
    return ", ".join(parts)


def render_markdown(summary: dict[str, Any]) -> str:
    """The summary as Markdown: sources, steps, examples of dropped rows, output shape."""
    out = summary["output"]
    lines = [
        f"# Build report: {out.get('path', '?')}",
        "",
        f"{_n(out.get('records'))} rows · SHA-256 `{out.get('sha256', '?')}` · "
        f"convmerge {summary.get('convmerge') or '?'} · recipe `{summary['recipe']}`",
        "",
        "## Sources",
        "",
        "| Source | From | License | Read | Converted | In the mix |",
        "|---|---|---|---:|---:|---:|",
    ]
    for s in summary["sources"]:
        lic = s["license"] or "unknown"
        if s["warning"]:
            lic += " ⚠"
        lines.append(
            f"| {_cell(s['name'])} | {_cell(s['origin'])} | {_cell(lic)} | "
            f"{_n(s['read'])} | {_n(s['converted'])} | {_n(s['mixed'])} |"
        )
    if summary["license_warnings"]:
        lines += ["", "**License warnings**", ""]
        lines += [f"- {w}" for w in summary["license_warnings"]]
    lines += ["", "## Steps", "", "| Step | In | Out | Removed | Why |", "|---|---:|---:|---:|---|"]
    for st in summary["steps"]:
        lines.append(
            f"| {_cell(st['name'])} | {_n(st['rows_in'])} | {_n(st['rows_out'])} | "
            f"{_n(st['removed'])} | {_cell(_why(st))} |"
        )
    examples = [(st["name"], s) for st in summary["steps"] for s in st["samples"]]
    if examples:
        lines += ["", "### Examples of removed rows", ""]
        for name, s in examples:
            text = f": {_cell(s['text'])}" if s.get("text") else ""
            lines.append(f"- {name} · {s['reason']} · line {_n(s.get('line'))}{text}")
    shape = summary.get("shape")
    if shape:
        lines += [
            "",
            "## Output",
            "",
            f"{_n(shape['rows'])} rows, layout `{shape['layout']}`"
            + (f", {_n(shape['pairs'])} preference pairs" if shape["pairs"] else "")
            + (f", {_n(shape['unreadable'])} unreadable" if shape["unreadable"] else "")
            + ".",
            "",
            "| User turns | " + " | ".join(_TURN_BUCKETS) + " |",
            "|---|" + "---:|" * len(_TURN_BUCKETS),
            "| Rows | " + " | ".join(_n(shape["turns"][b]) for b in _TURN_BUCKETS) + " |",
            "",
            "| Final answer (characters) | " + " | ".join(shape["answer_histogram"]) + " |",
            "|---|" + "---:|" * len(shape["answer_histogram"]),
            "| Rows | " + " | ".join(_n(v) for v in shape["answer_histogram"].values()) + " |",
            "",
            "Final answer length: p10 {p10}, median {p50}, p90 {p90}, max {max} characters.".format(
                **{k: _n(v) for k, v in shape["answer_chars"].items()}
            ),
        ]
    return "\n".join(lines) + "\n"


# --- HTML -----------------------------------------------------------------------

_CSS = """
:root{--bg:#fff;--fg:#1c1c1e;--muted:#6b6b70;--line:#e3e3e6;--bar:#3b6fd8;--warn:#b25b00}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#ececee;--muted:#9a9aa0;
--line:#2e2e32;--bar:#6d97ef;--warn:#f0a050}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;
max-width:960px;padding:24px 16px}
h1{font-size:1.4rem;margin:0 0 4px}h2{font-size:1.1rem;margin:28px 0 8px}
.meta{color:var(--muted);font-size:.9rem;overflow-wrap:anywhere}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.warn{color:var(--warn)}.bar{background:var(--line);height:8px;border-radius:4px;min-width:80px}
.bar>span{background:var(--bar);display:block;height:8px;border-radius:4px}
code{font-size:.85em;overflow-wrap:anywhere}td code,td.nw{white-space:nowrap}li{margin:2px 0}
""".strip()


def _e(value: Any) -> str:
    return html.escape(_n(value) if not isinstance(value, str) else value)


def _bar(part: int | None, whole: int | None) -> str:
    share = part / whole if part is not None and whole else 0
    return f'<div class="bar"><span style="width:{min(share, 1) * 100:.1f}%"></span></div>'


def render_html(summary: dict[str, Any]) -> str:
    """The summary as one self-contained HTML page (inline CSS, no scripts)."""
    out = summary["output"]
    title = f"Build report: {out.get('path', '?')}"
    body = [
        f"<h1>{_e(title)}</h1>",
        f'<p class="meta">{_e(out.get("records"))} rows · SHA-256 <code>'
        f"{_e(out.get('sha256', '?'))}</code> · convmerge {_e(summary.get('convmerge') or '?')}"
        f" · recipe <code>{_e(summary['recipe'])}</code></p>",
        "<h2>Sources</h2>",
        '<div class="scroll"><table><tr><th>Source</th><th>From</th><th>License</th>'
        '<th class="n">Read</th><th class="n">Converted</th><th class="n">In the mix</th></tr>',
    ]
    for s in summary["sources"]:
        lic = _e(s["license"] or "unknown")
        if s["warning"]:
            lic = f'<span class="warn" title="{_e(s["warning"])}">{lic} ⚠</span>'
        body.append(
            f"<tr><td>{_e(s['name'])}</td><td><code>{_e(s['origin'])}</code></td>"
            f"<td class=nw>{lic}</td><td class=n>{_e(s['read'])}</td>"
            f"<td class=n>{_e(s['converted'])}</td><td class=n>{_e(s['mixed'])}</td></tr>"
        )
    body.append("</table></div>")
    if summary["license_warnings"]:
        body.append(
            '<ul class="warn">'
            + "".join(f"<li>{_e(w)}</li>" for w in summary["license_warnings"])
            + "</ul>"
        )
    body += [
        "<h2>Steps</h2>",
        '<div class="scroll"><table><tr><th>Step</th><th class="n">In</th><th class="n">Out</th>'
        '<th>Kept</th><th class="n">Removed</th><th>Why</th></tr>',
    ]
    for st in summary["steps"]:
        body.append(
            f"<tr><td>{_e(st['name'])}</td><td class=n>{_e(st['rows_in'])}</td>"
            f"<td class=n>{_e(st['rows_out'])}</td>"
            f"<td>{_bar(st['rows_out'], st['rows_in'])}</td>"
            f"<td class=n>{_e(st['removed'])}</td><td>{_e(_why(st))}</td></tr>"
        )
    body.append("</table></div>")
    examples = [(st["name"], s) for st in summary["steps"] for s in st["samples"]]
    if examples:
        body.append("<h2>Examples of removed rows</h2><ul>")
        for name, s in examples:
            text = f": {_e(s['text'])}" if s.get("text") else ""
            body.append(
                f"<li><b>{_e(name)}</b> · {_e(s['reason'])} · line {_e(s.get('line'))}{text}</li>"
            )
        body.append("</ul>")
    shape = summary.get("shape")
    if shape:
        body += [
            "<h2>Output</h2>",
            f'<p class="meta">{_e(shape["rows"])} rows, layout <code>{_e(shape["layout"] or "?")}'
            f"</code>"
            + (f", {_e(shape['pairs'])} preference pairs" if shape["pairs"] else "")
            + (f", {_e(shape['unreadable'])} unreadable" if shape["unreadable"] else "")
            + "</p>",
        ]
        for caption, counts in (
            ("User turns per row", shape["turns"]),
            ("Final answer length (characters)", shape["answer_histogram"]),
        ):
            whole = max(counts.values()) or 1
            body.append(
                f'<div class="scroll"><table><tr><th>{_e(caption)}</th>'
                '<th class="n">Rows</th><th></th></tr>'
            )
            body += [
                f"<tr><td>{_e(k)}</td><td class=n>{_e(v)}</td><td>{_bar(v, whole)}</td></tr>"
                for k, v in counts.items()
            ]
            body.append("</table></div>")
        a = shape["answer_chars"]
        body.append(
            f'<p class="meta">p10 {_e(a["p10"])}, median {_e(a["p50"])}, '
            f"p90 {_e(a['p90'])}, max {_e(a['max'])} characters.</p>"
        )
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>\n"
        + "\n".join(body)
        + "\n</body></html>\n"
    )


# --- Dataset card ---------------------------------------------------------------


def dataset_card(summary: dict[str, Any], *, title: str | None = None) -> str:
    """A dataset card (``README.md``) for the output, with Hub front matter.

    The front matter's ``license`` is the sources' license when they share one,
    else ``other``. The card names every source with its license and says how
    to rebuild the file; nothing in it is uploaded anywhere.
    """
    out = summary["output"]
    records = out.get("records") or 0
    known = {s["license"] for s in summary["sources"] if s["license"]}
    unknown = any(not s["license"] for s in summary["sources"])
    license = next(iter(known)) if len(known) == 1 and not unknown else "other"
    size = next((label for top, label in _SIZES if records < top), "n>1B")
    name = title or Path(str(out.get("path", "dataset"))).stem
    shape = summary.get("shape") or {}
    lines = [
        "---",
        f"license: {license}",
        "size_categories:",
        f"- {size}",
        "tags:",
        "- convmerge",
        "---",
        "",
        f"# {name}",
        "",
        f"{_n(records)} rows built with convmerge {summary.get('convmerge') or '?'} from "
        f"{len(summary['sources'])} source(s) by the recipe `{summary['recipe']}`"
        + (f", in the `{shape['layout']}` layout" if shape.get("layout") else "")
        + ".",
        "",
        "## Sources",
        "",
        "| Source | From | License | Rows after conversion |",
        "|---|---|---|---:|",
    ]
    for s in summary["sources"]:
        lines.append(
            f"| {_cell(s['name'])} | {_cell(s['origin'])} | "
            f"{_cell(s['license'] or 'unknown')} | {_n(s['converted'])} |"
        )
    lines += ["", "## License", ""]
    if license == "other":
        lines.append(
            "The sources do not share one known license. Each row stays under the "
            "terms of the source it came from; check them before use."
        )
    else:
        lines.append(f"Every source is declared or carded as `{license}`.")
    if summary["license_warnings"]:
        lines += ["", *[f"- ⚠ {w}" for w in summary["license_warnings"]]]
    lines += [
        "",
        "## How it was built",
        "",
        "| Step | In | Out | Removed | Why |",
        "|---|---:|---:|---:|---|",
    ]
    for st in summary["steps"]:
        lines.append(
            f"| {_cell(st['name'])} | {_n(st['rows_in'])} | {_n(st['rows_out'])} | "
            f"{_n(st['removed'])} | {_cell(_why(st))} |"
        )
    lines += [
        "",
        "## Rebuild",
        "",
        "```bash",
        f"convmerge run {summary['recipe']}",
        "```",
        "",
        f"`{summary['lock']}` pins every input by content hash. The output's SHA-256 is "
        f"`{out.get('sha256', '?')}`; `convmerge run --frozen` fails when the files no "
        "longer match the lock.",
    ]
    return "\n".join(lines) + "\n"
