"""Documentation accuracy: do the commands, flags, recipes, and API names exist?

- every ``convmerge <command> ... --flag`` in README.md and docs/**/*.md is
  checked against the real argparse parser;
- every YAML block that looks like a recipe (has ``sources:``) is parsed with
  ``parse_recipe``; preset and manifest blocks are skipped;
- every ``convmerge.<name>`` / ``from convmerge import <name>`` in the docs
  must resolve.

Evaluation harness, not part of the package.
"""

from __future__ import annotations

import argparse
import importlib
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parser_flags() -> dict[str, set[str]]:
    from convmerge.cli import _build_parser

    top = _build_parser()
    flags: dict[str, set[str]] = {}
    for action in top._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                opts = {o for a in sub._actions for o in a.option_strings}
                for a in sub._actions:  # nested subcommands (preset init / validate)
                    if isinstance(a, argparse._SubParsersAction):
                        for sname, ssub in a.choices.items():
                            opts |= {o for x in ssub._actions for o in x.option_strings}
                            opts.add(sname)
                flags[name] = opts
    return flags


def doc_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def commands_in(text: str) -> list[str]:
    out: list[str] = []
    for block in re.findall(r"```(?:bash|sh|shell|console)?\n(.*?)```", text, re.S):
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.strip().lstrip("$ ").split(" #")[0]
            if line.startswith("convmerge "):
                out.append(line)
    out += re.findall(r"`(convmerge [a-z-]+ [^`]*)`", text)
    return out


def check_commands(flags: dict[str, set[str]]) -> list[str]:
    problems = []
    for path in doc_files():
        for line in commands_in(path.read_text(encoding="utf-8")):
            try:
                words = shlex.split(line)
            except ValueError:
                continue
            if len(words) < 2 or words[1].startswith("-"):
                continue
            cmd = words[1]
            if cmd not in flags:
                problems.append(f"{path.relative_to(ROOT)}: unknown command: {line[:90]}")
                continue
            for w in words[2:]:
                flag = w.split("=")[0]
                if flag.startswith("--") and flag not in flags[cmd]:
                    problems.append(f"{path.relative_to(ROOT)}: `{cmd}` has no {flag}: "
                                    f"{line[:90]}")  # fmt: skip
    return problems


def check_recipes() -> tuple[int, list[str]]:
    import yaml

    from convmerge.recipe import RecipeError, parse_recipe

    checked, problems = 0, []
    for path in doc_files():
        for block in re.findall(r"```ya?ml\n(.*?)```", path.read_text(encoding="utf-8"), re.S):
            if not re.search(r"^sources:", block, re.M) or "output:" not in block:
                continue
            try:
                raw = yaml.safe_load(block)
            except yaml.YAMLError as e:
                problems.append(f"{path.relative_to(ROOT)}: invalid YAML: {e}"[:160])
                continue
            checked += 1
            try:
                parse_recipe(raw, path=ROOT / "recipe.yaml")
            except RecipeError as e:
                problems.append(f"{path.relative_to(ROOT)}: {e}"[:200])
    return checked, problems


def check_api_names() -> list[str]:
    problems = []
    for path in doc_files():
        text = path.read_text(encoding="utf-8")
        names = set(re.findall(r"from convmerge import ([A-Za-z_, ]+)", text))
        for group in names:
            for name in (n.strip() for n in group.split(",")):
                if name and not hasattr(importlib.import_module("convmerge"), name):
                    problems.append(f"{path.relative_to(ROOT)}: convmerge has no {name}")
    return problems


def main() -> int:
    flags = parser_flags()
    cmd_problems = check_commands(flags)
    n_cmds = sum(len(commands_in(p.read_text(encoding="utf-8"))) for p in doc_files())
    n_recipes, recipe_problems = check_recipes()
    api_problems = check_api_names()
    print(f"doc files: {len(doc_files())}; command lines checked: {n_cmds}; "
          f"recipes parsed: {n_recipes}")  # fmt: skip
    for p in cmd_problems + recipe_problems + api_problems:
        print("PROBLEM", p)
    print(f"problems: {len(cmd_problems)} command/flag, {len(recipe_problems)} recipe, "
          f"{len(api_problems)} API")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
