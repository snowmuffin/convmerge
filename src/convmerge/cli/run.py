"""run command: execute a declarative recipe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _add_run(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "run",
        help="Run a recipe: fetch/normalize/convert each source, then mix and dedupe, "
        "repeating only what changed (YAML recipes need convmerge[preset])",
    )
    p.add_argument("recipe", nargs="?", type=Path, help="Recipe YAML/JSON file")
    p.add_argument(
        "--plan", action="store_true", help="Show which steps would run and why; do nothing"
    )
    p.add_argument(
        "--frozen",
        action="store_true",
        help="Fail (exit 1) if any step would run, i.e. the outputs are not exactly "
        "what the lock file records",
    )
    p.add_argument(
        "--force",
        nargs="*",
        action="extend",
        default=None,
        metavar="STEP",
        help="Re-run steps even if up to date: step names (tools.convert), source "
        "names, or kinds (fetch, convert, mix, ...); no value = everything",
    )
    p.add_argument("--hf-token", default=None)
    p.add_argument("--github-token", default=None)
    p.add_argument("--init", action="store_true", help="Write a commented recipe template")
    p.add_argument(
        "-o", "--output", type=Path, default=None, help="--init: file to write (default stdout)"
    )


def _unknown_force(recipe: Any, force: list[str] | None) -> str | None:
    """Say which ``--force`` names match no step, source, or kind (a typo would
    otherwise force nothing)."""
    import difflib

    from convmerge.recipe.engine import build_steps

    if not force:
        return None
    steps = build_steps(recipe)
    known = {n for s in steps for n in (s.name, s.kind, s.source) if n}
    missing = [f for f in force if f not in known]
    if not missing:
        return None
    hints = []
    for name in missing:
        close = difflib.get_close_matches(name, sorted(known), n=1)
        hints.append(f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    return (
        f"--force: no step, source, or kind named {', '.join(hints)}; "
        f"steps: {', '.join(s.name for s in steps)}"
    )


def _cmd_run(args: argparse.Namespace) -> None:
    from convmerge.recipe import (
        RECIPE_TEMPLATE,
        RecipeError,
        RecipeRunError,
        load_recipe,
        plan,
        run,
    )

    if args.init:
        if args.output:
            from convmerge.cli._common import refuse_existing

            refuse_existing(args.output)
            args.output.write_text(RECIPE_TEMPLATE, encoding="utf-8")
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            print(RECIPE_TEMPLATE, end="")
        return
    if args.recipe is None:
        print("error: a recipe file is required (or --init)", file=sys.stderr)
        sys.exit(2)
    if not Path(args.recipe).is_file():
        print(f"error: recipe file not found: {args.recipe}", file=sys.stderr)
        sys.exit(2)
    try:
        recipe = load_recipe(args.recipe)
    except (RecipeError, OSError, ImportError, ValueError) as e:
        print(f"error: {args.recipe}: {e}", file=sys.stderr)
        sys.exit(2)

    unknown = _unknown_force(recipe, args.force)
    if unknown:
        print(f"error: {unknown}", file=sys.stderr)
        sys.exit(2)

    if args.plan or args.frozen:
        try:
            steps = plan(recipe, force=args.force)
        except RecipeError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        width = max(len(s.step.name) for s in steps)
        for s in steps:
            print(f"{s.action:5} {s.step.name:{width}}  {s.reason}")
        pending = [s for s in steps if s.action != "skip"]
        if args.frozen and pending:
            print(
                f"error: {len(pending)} step(s) are not up to date with {recipe.lock_path.name}",
                file=sys.stderr,
            )
            sys.exit(1)
        return

    try:
        result = run(
            recipe, force=args.force, hf_token=args.hf_token, github_token=args.github_token
        )
    except RecipeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except RecipeRunError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    out = result.report["output"]
    print(
        f"[done] ran {len(result.ran)}, skipped {len(result.skipped)} -> {out['path']} "
        f"({out['records']:,} records); lock: {recipe.lock_path.name}, "
        f"report: {recipe.report_path.name}",
        file=sys.stderr,
    )


def _add_report(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "report",
        help="Render a recipe's last run as readable Markdown or HTML, or write a dataset "
        "card for its output (nothing is uploaded)",
    )
    p.add_argument("recipe", type=Path, help="Recipe YAML/JSON file that has been run")
    p.add_argument("--html", type=Path, default=None, metavar="PATH",
                   help="Write a self-contained HTML page here")  # fmt: skip
    p.add_argument("--markdown", type=Path, default=None, metavar="PATH",
                   help="Write Markdown here (default: print it if no file is named)")  # fmt: skip
    p.add_argument("--card", type=Path, default=None, metavar="PATH",
                   help="Write a dataset card (README.md with Hub front matter) here")  # fmt: skip
    p.add_argument("--title", default=None, help="--card: dataset name (default: output file stem)")
    p.add_argument("--no-scan", action="store_true",
                   help="Skip reading the output for turn and length distributions")  # fmt: skip


def _cmd_report(args: argparse.Namespace) -> None:
    from convmerge._output import atomic_text_writer
    from convmerge.io import refuse_overwrite
    from convmerge.recipe import RecipeError, load_recipe
    from convmerge.report import dataset_card, render_html, render_markdown, summarize_run

    if not Path(args.recipe).is_file():
        print(f"error: recipe file not found: {args.recipe}", file=sys.stderr)
        sys.exit(2)
    try:
        recipe = load_recipe(args.recipe)
    except (RecipeError, OSError, ImportError, ValueError) as e:
        print(f"error: {args.recipe}: {e}", file=sys.stderr)
        sys.exit(2)
    outputs = [args.html, args.markdown, args.card]
    refuse_overwrite([args.recipe, recipe.output, recipe.report_path, recipe.lock_path], outputs)
    try:
        summary = summarize_run(recipe, scan=not args.no_scan)
    except FileNotFoundError:
        print(f"error: no report at {recipe.report_path}; run the recipe first "
              f"(convmerge run {args.recipe})", file=sys.stderr)  # fmt: skip
        sys.exit(1)
    if not any(outputs):
        print(render_markdown(summary), end="")
        return
    for path, text in ((args.html, lambda: render_html(summary)),
                       (args.markdown, lambda: render_markdown(summary)),
                       (args.card, lambda: dataset_card(summary, title=args.title))):  # fmt: skip
        if path is None:
            continue
        with atomic_text_writer(path) as f:
            f.write(text())
        print(f"wrote {path}", file=sys.stderr)
