"""Declarative recipes: ``convmerge run recipe.yaml``.

A recipe chains existing commands (fetch → normalize → convert per source,
then mix and dedupe), records what it did in a lock file, and on the next run
repeats only the steps whose options, inputs, or convmerge version changed.
See ``docs/recipes.md``.
"""

from __future__ import annotations

from convmerge.recipe.engine import (
    PlannedStep,
    RecipeRunError,
    RunResult,
    Step,
    build_steps,
    load_lock,
    plan,
    run,
)
from convmerge.recipe.schema import Recipe, RecipeError, load_recipe, parse_recipe

__all__ = [
    "RECIPE_TEMPLATE",
    "PlannedStep",
    "Recipe",
    "RecipeError",
    "RecipeRunError",
    "RunResult",
    "Step",
    "build_steps",
    "load_lock",
    "load_recipe",
    "parse_recipe",
    "plan",
    "run",
]

RECIPE_TEMPLATE = """\
# convmerge recipe (version 1) — run with: convmerge run recipe.yaml
# Paths are relative to this file. See docs/recipes.md.
version: 1
workdir: build                  # intermediates: build/<source>/{raw,jsonl,converted.jsonl}
output: train/mixed.jsonl       # final training file
# lock: recipe.lock.json        # default: <recipe name>.lock.json next to this file
# report: build/report.json     # default: <workdir>/report.json

# auth:                         # same fields as a fetch manifest's auth block
#   hf_token_env: HF_TOKEN
#   github_token_env: GITHUB_TOKEN

sources:
  alpaca:
    # Exactly one of `fetch` (same fields as a fetch manifest entry, or
    # {manifest: path, name: entry}) or `path` (a local file or directory).
    fetch:
      url: https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
    # normalize: true             # default; false if the data is already clean JSONL
    convert: { from: alpaca, format: messages }

  local_chat:
    path: data/chat.jsonl
    convert:
      from: auto
      # preset: presets/chat.yaml
      # preference: chosen        # DPO / reward data: train on the chosen answer
      # on_invalid: drop          # drop | keep | fail
      # workers: 4                # parallel convert (does not change the output)

mix:                            # optional with one source; merge-all if omitted
  total: 50000
  seed: 42
  weights: { alpaca: 0.7, local_chat: 0.3 }

dedupe: true                    # or { keys: [messages], algorithm: md5 }
"""
