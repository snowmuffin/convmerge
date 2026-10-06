# Recipes (`convmerge run`)

A recipe describes a whole training mix in one file: where each source
comes from, how it is converted, and how the sources are mixed and
deduplicated. `convmerge run` executes it, writes a **lock file** recording
exactly what it did, and on the next run repeats only what changed.

```bash
convmerge run --init -o recipe.yaml   # commented template
convmerge run recipe.yaml --plan      # what would run, and why
convmerge run recipe.yaml             # do it
convmerge run recipe.yaml --frozen    # exit 1 unless everything is up to date
```

Every step is an existing command called with the same options, so a recipe
produces byte-for-byte the same files as typing the commands by hand. YAML
recipes need PyYAML (`convmerge[preset]` or `[all]`); JSON recipes do not.

## Example

```yaml
version: 1
workdir: build                    # intermediates (default: build)
output: train/mixed.jsonl         # the final training file

sources:
  alpaca:
    path: data/alpaca_data.json   # a local file or directory
    convert: { from: alpaca, workers: 4 }

  tools:
    fetch:                        # same fields as a fetch manifest entry
      url: https://raw.githubusercontent.com/org/repo/main/tools.jsonl
      max_rows: 50000
    convert: { from: sharegpt, keep_meta: [source] }

  prefs:
    fetch: { manifest: manifest.yaml, name: dpo-en }   # reuse a manifest entry
    convert: { from: auto, preference: chosen }

mix:
  total: 100000
  seed: 42
  weights: { alpaca: 0.6, tools: 0.3, prefs: 0.1 }

dedupe: { keys: [messages] }
```

All paths are relative to the recipe file.

## Reference

### Top level

| Key | Default | Meaning |
|-----|---------|---------|
| `version` | `1` | Recipe format version. |
| `output` | *(required)* | Final JSONL file. |
| `workdir` | `build` | Intermediates go to `<workdir>/<source>/{raw, jsonl, converted.jsonl}`. |
| `lock` | `<recipe name>.lock.json` | Lock file, next to the recipe. Commit it to share exact builds. |
| `report` | `<workdir>/report.json` | Per-step stats, drop reasons, the output digest, and every source's license (`licenses`, `license_warnings`). |
| `auth` | — | Token sources, same fields as a fetch manifest's `auth` block. |
| `sources` | *(required)* | Name → source. Names become directory names (letters, digits, `_ - .`). |
| `mix` | — | How to combine sources (see below). |
| `dedupe` | — | `true`, or `{keys: [...], algorithm: md5 \| sha256}`; `near: true` removes near-duplicates instead (see below). |
| `filter` | — | `true` (default rules), or rule options: drop refusals, empty answers, loops, bad pairs (see below). |
| `decontam` | — | Drop rows that overlap evaluation sets (see below). |
| `tokens` | — | Keep rows that render with a model's chat template and fit a length (see below). Needs `convmerge[tokens]`. |
| `split` | — | Write a validation set next to the output (see below). |

### Sources

Each source has exactly one of:

- `path` — a local file or directory (directories are walked like
  `convmerge normalize`).
- `fetch` — either the fields of a [fetch manifest](fetch.md) entry
  (`hf`, `url`, `split`, `config`, `ext`, `mode`, `lfs`, `max_rows`), or
  `{manifest: path, name: entry}` to reuse an entry of an existing manifest
  (optionally with `max_rows` to sample it). A manifest's `auth` block comes
  along with its entries.

and then:

| Key | Default | Meaning |
|-----|---------|---------|
| `license` | the dataset card's | The source's license, recorded in the report. Hugging Face sources without one get the `license` of their dataset card at fetch time. Non-commercial, research-only, `other`, and unknown licenses are warned about (`[license]` lines). |
| `normalize` | `true` | Normalize before converting; `false` if the data is already clean JSONL; `{array_key: ...}` to rename the wrapper for array records; `{sheet: ...}` to read that sheet of an `.xlsx` path (default: the first). |
| `convert.from` | — | Adapter (`alpaca`, `sharegpt`, `chat`/`auto`, or a plugin). Required unless a preset sets it. |
| `convert.format` | `messages` | Output format: `messages`, `alpaca`, or `preference` (DPO pairs; mix and dedupe work on them as on any JSONL). |
| `convert.preset` | — | A [preset](custom_presets.md) file; its contents are part of the step's inputs, so editing it re-runs the step. |
| `convert.adapter_kwargs` | — | Same as `--adapter-kwargs`, as a mapping. |
| `convert.preference` | — | `chosen` / `rejected` for preference data. |
| `convert.on_invalid` | `drop` | `drop`, `keep`, or `fail`. |
| `convert.workers` | `1` | Parallel convert. Output does not depend on it, so changing it never re-runs a step. |
| `convert.tool_arguments`, `keep_meta`, `meta_key`, `alpaca_multiturn`, `reasoning`, `tool_content`, `train_turns`, `media` | — | Same as the `convert` flags. |
| `convert.meta` | — | Constant fields written under `meta` on every row (`--meta`), e.g. `{dataset: kullm}` to keep each row's origin after the mix. |
| `convert.map` | — | A [field mapping](format.md#field-mapping---from-map) for layouts no adapter knows; implies `from: map`. |
| `convert.system`, `merge_consecutive`, `split_turns`, `reasoning_turns`, `leading_assistant` | — | The template fixes of `convert` (`--system`, `--merge-consecutive`, `--split-turns`, `--reasoning-turns`, `--leading-assistant`); see [format.md](format.md#fixes-for-strict-chat-templates). |

A directory source with several data files is converted file by file (sorted
by path) and concatenated.

### Mix and dedupe

| Key | Default | Meaning |
|-----|---------|---------|
| `mix.weights` | equal | Weight per source; every source needs one. |
| `mix.total` | all records | Target number of records. |
| `mix.seed` | `42` | |
| `mix.oversample` | `false` | Repeat records of sources smaller than their share. |
| `mix.sampler` | `v2` | `v1` reproduces pre-0.7 mixes. |
| `mix.by` | `rows` | What the weights measure: `rows`, `chars`, or `tokens` (needs `total`; `tokens` needs `tokenizer`). `total` stays a row count, split so each source's share of the characters or tokens is its weight. |
| `mix.tokenizer` | — | Hugging Face tokenizer name or path for `by: tokens` (needs `convmerge[tokens]`). |
| `mix.by_sample` | — | With `by: chars` / `tokens`: measure this many random rows per source and scale, instead of every row (an estimate; see `mix --by-sample`). |
| `mix.max_tokens` | — | With `by: tokens`: count each row as at most this many tokens, the trainer's maximum length, so the weights describe the text that is trained on (see `mix --max-tokens`). |

| `dedupe.near` | `false` | Drop rows whose text is a near-copy of an earlier row (MinHash LSH over word 5-grams). Needs `convmerge[quality]`. |
| `dedupe.threshold` | `0.8` | `near`: estimated Jaccard similarity that makes a duplicate (below 1). |
| `dedupe.num_perm` | `128` | `near`: MinHash permutations. |
| `dedupe.workers` | `1` | `near`: processes computing MinHashes. Output does not depend on it (1.1+). |

With several sources and no `mix` block, every record of every source is
merged and shuffled (seed 42). With one source and no `mix`, the converted
file goes straight to `dedupe` or `output`.

### Filter and decontam

These run the `filter` and `decontam` commands ([quality.md](quality.md))
on the mixed file.

| Key | Default | Meaning |
|-----|---------|---------|
| `filter.enable` / `filter.disable` | — | Rule names to turn on or off (defaults: `empty_answer`, `refusal`, `repetition`, `near_identical_pair`, `rejected_empty`). |
| `filter.min_answer_chars` | `1` | `empty_answer`: shortest final answer kept. |
| `filter.min_chars` / `filter.max_chars` | — | Turn on `length`: bounds on the answer text. |
| `filter.repetition_max` | `0.7` | `repetition`: share of repeated word 10-grams that marks a loop. |
| `filter.slop_max` | `3` | `slop`: stock phrases per row that drop it. |
| `filter.min_script` | — | Turn on `script`, e.g. `{hangul: 0.3}`. |
| `filter.rules_file` | — | Extra phrases and regex rules (a step input). |
| `filter.workers` | `1` | Processes checking rows. Output does not depend on it (1.1+). |
| `decontam.against` | *(required)* | Evaluation sets: JSONL files (relative to the recipe; step inputs) or `hf:REPO[:CONFIG[:SPLIT]]` (split defaults to `test`). |
| `decontam.ngram` | `13` | Words per n-gram. |
| `decontam.min_tokens` | `8` | Shorter evaluation passages must appear whole; shorter still are skipped. |
| `decontam.check` | `prompts` | `prompts` (system and user turns) or `all`. |
| `decontam.fields` | every string field | Evaluation fields to read. |

```yaml
filter:
  enable: [slop]
  min_script: {hangul: 0.3}      # Korean mixes: drop half-translated rows
decontam:
  against: [hf:openai/gsm8k:main, hf:cais/mmlu:all, evals/internal.jsonl]
```

### Tokens and split

Stages run in this order: sources → `mix` → `dedupe` → `filter` →
`decontam` → `tokens` → `split`.

| Key | Default | Meaning |
|-----|---------|---------|
| `tokens.tokenizer` | *(required)* | Hub model name, or a tokenizer directory relative to the recipe (its files are step inputs). |
| `tokens.max_tokens` | — | Drop rows longer than this (a preference pair counts as its longer side). |
| `tokens.revision` | — | Tokenizer revision; pin it for fully reproducible builds. |
| `tokens.chat_template` | the tokenizer's | A Jinja file to render with instead (a step input). |
| `split.val` / `split.val_rows` | — | One of: a fraction of rows (hash-based, about that many) or an exact count; `val_rows` errors if its boundary would split a duplicate/`keys` group. |
| `split.seed` | `42` | |
| `split.keys` | whole row | Hash only these top-level keys, so rows sharing them stay on one side. |
| `split.val_output` | `<output stem>.val.jsonl` | Where the validation rows go; `output` gets the rest. |

`tokens` drops rows the chat template rejects and rows over `max_tokens`
(the report lists both, with the length distribution, and the warnings and
`hints` described under `convmerge tokens` in the README). `split` assigns rows
by content hash, so duplicates never straddle train and validation; it runs
as two steps, `split.train` and `split.val`, that re-run independently.

## How re-runs are decided

Steps run in order — per source `fetch`/`normalize`/`convert`, then `mix`,
`dedupe`, `filter`, `decontam`, `tokens`, `split`, and `output` (a copy, when no other step writes the final file).
The lock file records, per step, the options, the convmerge version, the
SHA-256 of every input, and the SHA-256 of the output. A step runs when:

- it never ran, or `--force` selects it;
- its options or the convmerge version changed;
- an input changed (a local source file, a preset, or the output of the step
  before it);
- its output is missing or was modified by hand.

Otherwise it is skipped. If a step re-runs and produces the **same** output,
the steps after it are still skipped. Digests are cached by file size and
modification time, so unchanged multi-GB files are not re-hashed.

Remote data is not re-downloaded just because a run starts: a `fetch` step
re-runs when its options change. Use `--force fetch` to refresh remote
sources — if the download is identical, nothing downstream re-runs. The
same holds for `decontam`'s `hf:` evaluation sets: they are downloaded when
the step runs, and `--force decontam` picks up a changed Hub dataset.

`--force` accepts step names (`tools.convert`), source names (`tools` — all
of its steps), or kinds (`fetch`, `normalize`, `convert`, `mix`, `dedupe`);
`--force` alone re-runs everything.

Each step writes to a hidden `.part` path and is renamed into place only
when it succeeds, so a failed run leaves the previous outputs and the lock
file intact.

## Using recipes in CI

Commit the recipe and the lock file. `convmerge run recipe.yaml --frozen`
exits 1 (and prints the plan) when any step is not up to date — for example,
when someone changed a preset without rebuilding. `report.json` holds the
drop reasons per source and the output's SHA-256 for review.

## Python API

```python
from convmerge.recipe import load_recipe, plan, run

recipe = load_recipe("recipe.yaml")
for step in plan(recipe):
    print(step.action, step.step.name, step.reason)
result = run(recipe, force=["tools"])
print(result.ran, result.report["output"])
```

### Protecting recipe files and outputs

Since 1.6.5, the complete recipe path graph is checked before any step runs. Keep
source/configuration paths separate from the work directory, output, lock and
report. In particular, `report: train.jsonl` cannot coexist with
`output: train.jsonl`, and the final output cannot be a source or the recipe
itself. Links and overlapping directory trees are checked as well. See
[output safety](output-safety.md) for the exact boundaries.

A malformed lock is an explicit error, not an instruction to discard cached
state. Back up and remove that lock to request a full rebuild.
