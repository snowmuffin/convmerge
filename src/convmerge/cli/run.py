"""run command: execute a declarative recipe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
            args.output.write_text(RECIPE_TEMPLATE, encoding="utf-8")
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            print(RECIPE_TEMPLATE, end="")
        return
    if args.recipe is None:
        print("error: a recipe file is required (or --init)", file=sys.stderr)
        sys.exit(2)
    try:
        recipe = load_recipe(args.recipe)
    except (RecipeError, OSError, ImportError, ValueError) as e:
        print(f"error: {args.recipe}: {e}", file=sys.stderr)
        sys.exit(2)

    if args.plan or args.frozen:
        steps = plan(recipe, force=args.force)
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
