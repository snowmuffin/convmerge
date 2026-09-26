# Design: declarative recipes (`convmerge run`)

Status: **proposal** for 0.8 — not implemented. Comments welcome on the
tracking issue before any code lands.

## Problem

A real training mix is several commands per source, then a few over all of
them:

```
fetch → normalize → convert (per-source preset) ─┐
fetch → normalize → convert (per-source preset) ─┼→ mix → dedupe → train.jsonl
fetch → normalize → convert (per-source preset) ─┘
```

Today this lives in shell scripts. They are easy to get subtly wrong (a
stale intermediate file, a changed preset nobody re-ran), hard to review, and
the only reproducibility record is the `.mix.json` sidecar of the last step.

## What went wrong in 0.3.0

0.3.0 added `pipeline`, `reshape`, `resume`, `sample`, `merge`, `split`, and a
`build` command, and was reverted in 0.3.1 "pending more design iteration".
Whatever the details, its shape is the risk to avoid: **new primitives next
to the existing commands**, so the pipeline and the CLI could drift apart and
the surface area roughly doubled.

## Constraints for the new design

1. **No new transforms.** Every step is an existing command with the same
   options it has on the CLI (`fetch`, `normalize`, `convert`, `mix`,
   `dedupe`). A recipe step and a hand-typed command produce identical files.
2. **Thin orchestration.** `convmerge run` resolves paths, orders steps, skips
   what is up to date, and records what it did. It holds no data-processing
   logic of its own.
3. **Reproducible by default.** Every run writes `recipe.lock.json`: convmerge
   version, per-step options, input and output SHA-256 digests, seeds, and
   the fetch markers it relied on.
4. **Incremental.** A step re-runs only when its inputs' digests, its options,
   or the convmerge version changed — the same idea as the fetch
   completion markers, applied to every step.
5. **Inspectable.** `convmerge run --plan` prints the steps and why each would
   run or be skipped, without doing anything.

## Sketch

```yaml
version: 1
workdir: ./build            # intermediates: build/<source>/{raw,jsonl,converted}
sources:
  alpaca_ko:
    fetch: { hf: org/alpaca-ko, split: train }
    convert: { from: alpaca, format: messages }
  tools:
    fetch: { url: https://raw.githubusercontent.com/org/repo/main/tools.jsonl, max_rows: 50000 }
    convert: { preset: presets/tools.yaml, preference: chosen }
mix:
  total: 100000
  seed: 42
  weights: { alpaca_ko: 0.7, tools: 0.3 }
dedupe: { keys: [messages] }
output: ./train/mixed.jsonl
```

Each `sources.<name>` block maps one-to-one onto existing manifest entries and
convert options; `mix` and `dedupe` onto their commands.

## Open questions

- Per-source validation reports: merge into one run report, or keep one
  `--report` file per source?
- Should `run` accept an existing fetch manifest, so current users keep their
  `manifest.yaml` and only add the convert/mix parts?
- Concurrency: run independent sources in parallel, or leave that to
  `convert --workers` inside each step?
- Where lock files live, and whether `run --frozen` should refuse to run when
  anything differs from the lock (for CI).
