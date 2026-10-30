# convmerge

[![PyPI](https://img.shields.io/pypi/v/convmerge.svg)](https://pypi.org/project/convmerge/)
[![Python versions](https://img.shields.io/pypi/pyversions/convmerge.svg)](https://pypi.org/project/convmerge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/snowmuffin/convmerge/actions/workflows/ci.yml/badge.svg)](https://github.com/snowmuffin/convmerge/actions/workflows/ci.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/convmerge.svg)](https://pypi.org/project/convmerge/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

> **Convert Alpaca, ShareGPT, tool-calling, preference, and mixed chat datasets into one training-ready JSONL.**  
> Fetch from HuggingFace or GitHub, normalize messy Parquet / JSON / JSONL, convert 65 tested dataset layouts into `messages` (SFT) or `{prompt, chosen, rejected}` (DPO) rows, weighted-mix multiple domain sources, deduplicate, filter, and decontaminate — one command each, or the whole pipeline from a reproducible recipe.

`convmerge` is a **data-preparation CLI and library** for LLM supervised fine-tuning (SFT).
It takes heterogeneous instruction-tuning datasets — **Alpaca**, **ShareGPT**, raw chat JSONL,
template-rendered `text` columns, **tool-calling** data (OpenAI, Hermes, Glaive, xLAM,
LLaMA-Factory), **preference** data (UltraFeedback, HH-RLHF, Orca DPO pairs, Arena),
Parquet dumps — and produces a single clean JSONL file in the standard `messages` format
(or `alpaca` shape, or DPO preference pairs) that TRL, LLaMA-Factory, axolotl, and other
fine-tuning frameworks consume directly.

It is scoped to the step **before the training loop**: it does not train
models or orchestrate training jobs, and this release does not run models
either. See [Out of scope](#out-of-scope) below.

**Repository:** [github.com/snowmuffin/convmerge](https://github.com/snowmuffin/convmerge)  
**Status:** stable since 1.0: the CLI, the Python API, and the file formats
change incompatibly only in a new major version ([stability.md](docs/stability.md)).

## Quickstart

Three rows in three layouts (Alpaca, ShareGPT, OpenAI messages) become one
training file:

```bash
pip install convmerge
cat > raw.jsonl <<'JSONL'
{"instruction": "Translate to French", "input": "Good morning", "output": "Bonjour"}
{"conversations": [{"from": "human", "value": "What is 2+2?"}, {"from": "gpt", "value": "4"}]}
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
JSONL
convmerge convert -i raw.jsonl -o train.jsonl --from auto
# read 3 lines, wrote 3 examples
head -n 1 train.jsonl
# {"messages": [{"role": "user", "content": "Translate to French\nGood morning"}, {"role": "assistant", "content": "Bonjour"}]}
```

A dataset from the Hub (`fetch` needs the `[fetch-all]` extra):

```bash
pip install "convmerge[fetch-all]"
convmerge fetch hf://your-org/your-dataset --max-rows 1000 -o raw
convmerge convert -i raw/your-org_your-dataset.jsonl -o train.jsonl --from auto
```

A file saved in another encoding (cp949 on Korean Windows, for example) is
read with `--encoding cp949`; the output is always UTF-8.

## When it helps, and when it doesn't

convmerge is worth adding when **several datasets in different layouts** go into
one training run. Of the 65 public datasets convmerge is tested against, 10 load
into TRL's `SFTTrainer` / `DPOTrainer` as they are; the other 55 need a
conversion script of their own (ShareGPT roles, Alpaca columns, message trees,
tool calls in five encodings, scored or labeled preference pairs). Once there
are several sources, the steps after conversion come up too: weighting the mix
by text rather than rows, removing copies across sources, keeping evaluation
prompts out, and checking each row against the target model's chat template.
Each of those is one command here, or one recipe for the whole pipeline.

It adds little when:

- **You train on one dataset that is already in `messages` or `{prompt, chosen, rejected}`.**
  `datasets.load_dataset` plus the trainer is enough.
- **You need model-based cleaning:** quality classifiers, LLM judges, topic or
  safety labels. convmerge's `filter` is rule-based only (see
  [Out of scope](#out-of-scope)).
- **You process web-scale corpora.** convmerge streams line by line on one
  machine; billions of documents need a distributed pipeline.
- **You need personal information removed.** Use a dedicated tool first (see
  [Out of scope](#out-of-scope)).

Does cleaner data train a better model? A small pre-registered study
([docs/effect.md](docs/effect.md)) trained SmolLM2-135M on the same four
sources prepared by common scripts and by a convmerge recipe. It found no
behaviour difference, and a slightly lower eval loss for the scripts, which
put 1.7 times as many tokens into the same number of steps. convmerge saves
the conversion work and shows what the data contains. This study found no
sign that it makes a small model better.

Three complete recipes with real datasets are in
[examples/recipes](examples/recipes): a Korean SFT mix, a DPO mix of three
preference layouts, and a tool-calling mix checked against a Qwen chat template.

## Install

```bash
pip install convmerge                    # core: convert, dedupe, filter, decontam; normalize for .json/.jsonl
pip install "convmerge[all]"             # full CLI: fetch (HF+GitHub), parquet, YAML presets, tokens, near dedupe
```

Granular extras:

```bash
pip install "convmerge[fetch]"           # YAML manifests + GitHub (PyYAML)
pip install "convmerge[fetch-all]"       # fetch + HuggingFace (``datasets``)
pip install "convmerge[fetch-hf]"        # same dependencies as ``fetch-all`` (backward-compatible name)
pip install "convmerge[parquet]"         # Parquet input for ``normalize``
pip install "convmerge[preset]"          # YAML convert presets (`--preset`, `preset validate`)
pip install "convmerge[tokens]"          # `tokens`: lengths + chat-template checks (transformers, no PyTorch)
pip install "convmerge[quality]"         # `dedupe --near`: near-duplicate removal (datasketch)
pip install "convmerge[xlsx]"            # Excel (.xlsx) input for ``normalize`` (openpyxl)
```

| Command / feature | Extra |
|-------------------|--------|
| `convert`, `dedupe`, `filter`, `decontam` (local eval files), `turns`, `split`, `llamafactory-info`, `axolotl-config` | *(core)* |
| `normalize` on `.parquet` | `[parquet]` |
| `normalize` on `.xlsx` | `[xlsx]` |
| `fetch` with YAML manifest or GitHub | `[fetch]` |
| `fetch` with HuggingFace manifest entries | `[fetch-all]` or `[fetch-hf]` |
| `convert --preset`, `preset` | `[preset]` |
| `tokens` | `[tokens]` |
| `dedupe --near` | `[quality]` |
| `decontam --against hf:...` | `[fetch-all]` or `[fetch-hf]` |
| Everything above | `[all]` |

Or from a clone:

```bash
git clone https://github.com/snowmuffin/convmerge.git
cd convmerge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,all]"
```

## From datasets to a training run

| Trainer | Guide | Formats |
|---------|-------|---------|
| TRL (`SFTTrainer`, `DPOTrainer`) | [docs/guides/trl.md](docs/guides/trl.md) | `messages`, `preference` |
| LLaMA-Factory | [docs/guides/llamafactory.md](docs/guides/llamafactory.md) | `sharegpt`, `sharegpt-preference` + `llamafactory-info` |
| axolotl | [docs/guides/axolotl.md](docs/guides/axolotl.md) | `messages`, `sharegpt`, `preference` + `axolotl-config` |

Each guide is one recipe — fetch, convert, mix, dedupe, a `tokens` filter for
the target model's chat template and length, and a train/validation `split` —
plus the trainer config. All three were run end to end (SFT and DPO).
Korean data: [docs/guides/korean.md](docs/guides/korean.md) (한국어 가이드).

## Commands

### 1. `fetch` — pull raw data from HF + GitHub via a YAML manifest

> HuggingFace entries delegate to `datasets.load_dataset(...).to_json(...)`,
> i.e. the output is a **JSONL dump** of the selected split. GitHub entries
> support a single raw URL, recursive Trees API fetch with an extension
> filter, or `git clone` (with optional `git lfs pull`). `fetch` is a
> reproducible downloader, not a mirror of HuggingFace's Arrow cache.

```yaml
# manifest.yaml
version: 1
defaults: { output_root: ./raw, resume: true }
auth:     { hf_token_env: HF_TOKEN, github_token_env: GITHUB_TOKEN }
datasets:
  - { name: hub-sft, hf: your-org/your-dataset, split: train }
  - { name: orca-raw,
      url: https://raw.githubusercontent.com/org/repo/main/data/train.jsonl }
  - { name: repo-tree,
      url: https://github.com/org/example-repo, ext: [".jsonl"] }
  - { name: big-lfs,
      url: https://github.com/org/big-lfs-repo, mode: clone, lfs: true }
```

```bash
convmerge fetch manifest.yaml -o ./raw
# or one-shot shortcuts:
convmerge fetch hf://org/dataset -o ./raw --split train
convmerge fetch https://github.com/org/repo -o ./raw --ext .jsonl
```

Tokens resolve in order CLI flag → file → env var, and are redacted from logs.
See [docs/fetch.md](docs/fetch.md) for the full schema.

### 2. `normalize` — reshape parquet / messy JSON into clean JSONL

```bash
convmerge normalize -i ./raw -o ./jsonl
```

Handles parquet (streamed via `pyarrow`), top-level JSON arrays, concatenated
single-line JSON (`{...}{...}{...}`), JSONL whose lines are arrays (wrapped as
`{"conversation": [...]}`), and already-valid JSONL. A directory input is
walked recursively and mirrored under the output directory. A `.csv`,
`.tsv`, or `.xlsx` file given as `-i` becomes one object per row, keyed by the
header (`instruction,input,output` spreadsheets convert with `--from auto`
next; `.xlsx` needs `convmerge[xlsx]`, `--sheet NAME` picks a sheet).

File outputs are published only after a successful write. Failed files keep
their previous output; folder runs still commit independent successes and
return exit 1 if any file failed. See [output safety](docs/output-safety.md).

### 3. `convert` — adapter + emitter pipeline

```bash
convmerge convert -i ./jsonl/alpaca.jsonl -o ./train/alpaca.messages.jsonl \
  --from alpaca --format messages

convmerge convert -i ./jsonl/mixed.jsonl -o ./train/mixed.messages.jsonl \
  --from auto --format messages         # auto-detecting chat adapter

# Optional: YAML preset (pip install "convmerge[preset]")
convmerge preset init -o convert_preset.yaml
convmerge preset validate convert_preset.yaml
convmerge convert -i ./jsonl/mixed.jsonl -o ./out.jsonl --preset convert_preset.yaml
```

Adapters: `alpaca`, `sharegpt`, `chat` (alias `auto`), and `map` for any other
layout (dotted paths such as `dialogue[].utterances[]`, no code; see
[docs/format.md](docs/format.md#field-mapping---from-map)).  
Output formats: `messages`, `alpaca`, `preference` (DPO pairs), `sharegpt` and
`sharegpt-preference` (LLaMA-Factory / Unsloth).

> **Tool calling and multimodal:** OpenAI `tool_calls` / `tools`, LLaMA-Factory
> `function_call` / `observation` turns, Hermes `<tool_call>` tags, Glaive and
> xLAM layouts all come out as standard `tool_calls` / `tools`, and image /
> audio / video references (`image_url` parts, `images` columns with `<image>`
> tokens) are preserved in the `messages` output (`--media placeholders` writes
> TRL's vision-language layout: `{"type": "image"}` parts plus an `images`
> column). Media is kept by reference only — convmerge never
> downloads or decodes it. `--from sharegpt` keeps whole conversations since
> 0.6.0 (`turn_mode: pairs` restores the old split). See
> [docs/format.md](docs/format.md#sharegpt).

Preference (DPO / reward) datasets: `--format preference` writes
`{prompt, chosen, rejected}` pairs for TRL's `DPOTrainer`; `--preference chosen`
trains SFT on the chosen answer instead (UltraFeedback, HH-RLHF, Orca DPO pairs,
LLaMA-Factory ranking, TRL, and Chatbot Arena shapes).

Rendered `text` columns (ChatML, Llama 2/3, Gemma, `### Human:`, HH, the Alpaca
prompt) are split back into turns by `--from auto`.

Reasoning data: traces stored inline (`<think>...</think>`) or in a
`reasoning_content` / `thinking` field are kept; `--reasoning` moves them to
the field your model's chat template reads (`reasoning_content` for Qwen3 /
DeepSeek, `thinking` for gpt-oss), inline, or drops them. See
[docs/guides/reasoning.md](docs/guides/reasoning.md).

Strict chat templates: `--system fold` (no system role), `--merge-consecutive`
("roles must alternate"), `--reasoning-turns last` / `--split-turns`
(templates that render reasoning only after the last user turn). Tool-call
turns get `"content": ""`, which every common template accepts. Symptom →
fix table: [docs/guides/troubleshooting.md](docs/guides/troubleshooting.md).

Large files: `--workers N` converts with N processes (same output and
stats as a single process; ~3.8x faster with 4 workers in our benchmark).
`filter` and `dedupe --near` take `--workers N` too.

Every example is validated before it is written; ones with no user turn,
empty messages, or unmatched tool results are dropped and counted by reason
(`--on-invalid keep|fail` to change that, `--report PATH` for details,
`convmerge validate -i FILE` to check an existing file). See
[docs/format.md](docs/format.md#validation).

Presets and team-specific tuning: [docs/custom_presets.md](docs/custom_presets.md).

> `chat` / `auto` is a **heuristic** adapter: it inspects the keys of each
> input record (`messages`, `conversation(s)`, `text`, `conversation_a`/`_b`,
> `instruction`/`input`/`output`, …) and routes to the right branch with a
> configurable role map. For unusual schemas, pin an explicit adapter
> (`alpaca`, `sharegpt`) or override keys programmatically — see
> [docs/format.md](docs/format.md).

### 4. `mix` — domain-controlled weighted merge

```bash
# Inline weights
convmerge mix \
  -i ./train/code.messages.jsonl:0.4 \
     ./train/math.messages.jsonl:0.3 \
     ./train/general.messages.jsonl:0.3 \
  -o ./train/mixed.jsonl --total 100000 --seed 42

# Or via a config file (YAML requires convmerge[preset])
convmerge mix mix.yaml
```

```yaml
# mix.yaml
seed: 42
total: 100000
output: ./train/mixed.jsonl
sources:
  - { path: ./train/code.messages.jsonl,    weight: 0.4 }
  - { path: ./train/math.messages.jsonl,    weight: 0.3 }
  - { path: ./train/general.messages.jsonl, weight: 0.3 }
```

Weights are normalized automatically and need not sum to 1.0. When a source
has fewer records than its allocation it is clipped; pass `--oversample` to
repeat records instead. `mix` streams: it reads each source twice and shuffles
through temporary files next to the output, so memory stays small even when
merging multi-GB sources (`--sampler v1` reproduces mixes made before 0.7). A sidecar `.mix.json` is written alongside
the output recording the exact seed, weights, and per-source counts for full
reproducibility. Omit `--total` to merge all records from every source.

Weights count rows by default. Sources whose rows differ a lot in length
(short chat against long reasoning traces) can be weighted by text instead:
`--by chars`, or `--by tokens --tokenizer Qwen/Qwen2.5-7B-Instruct`, splits
the `--total` rows so each source's share of the characters or tokens is its
weight. `mix` prints each source's share, and the share of rows with a
reasoning trace, so the mix you get is the mix you meant. `--by chars` is
fast and enough when the sources share a language; `--by tokens` tokenizes
every row, so on large sources add `--by-sample 5000` to measure 5,000
random rows per source and scale (an estimate, the same for the same seed).
A source with too few rows is written whole and the others keep their
counts, so the mix moves off its weights; `mix` warns when a share ends more
than 5 points from its weight (`--oversample` repeats the short source).
Weights by text count whole rows, but trainers cut rows at their maximum
length. Pass that length as `--max-tokens N` with `--by tokens` (recipe:
`mix.max_tokens`) and each row counts as at most N tokens, so the weights
describe the text that is trained on; `mix` also prints each source's share
of rows longer than N. Without it a source of very long rows gets its share
on paper and loses most of it to truncation, as the
[effect study](docs/effect.md) found.

### 5. `dedupe` / `filter` / `decontam` / `tokens` / `split` — ready for training

```bash
convmerge dedupe -i ./train/mixed.jsonl -o ./train/mixed.dedup.jsonl
# --near also drops near-copies (MinHash LSH; needs convmerge[quality])

# Rule-based quality filter: refusals and "as an AI language model" disclaimers
# (English, Korean), empty answers, looping answers or reasoning traces, and
# preference pairs whose answers are the same or whose rejected side is empty.
# Optional: --enable slop, --min-chars/--max-chars, --min-script hangul=0.3.
# Prints a JSON report with example rows per rule; -o keeps the rows that pass.
convmerge filter -i ./train/mixed.dedup.jsonl -o ./train/clean.jsonl --rejects rejected.jsonl

# Drop rows that share a 13-word n-gram with a benchmark (prompts; --check all
# for answers too). hf: sources need convmerge[fetch-all].
convmerge decontam -i ./train/clean.jsonl --against hf:your-org/your-eval-set:main \
  --against ./evals/held_out.jsonl -o ./train/decontam.jsonl

# Render every row with the model's chat template: length percentiles, rows the
# template rejects (with line numbers), double-encoded tool arguments, answers
# that start beyond --max-tokens, answers not followed by a stop token, reasoning
# the template drops, {% generation %} support, and the instruction/response
# markers for Unsloth's train_on_responses_only -- with the convert flag that
# fixes each ("hints"). -o keeps the rows that render and fit.
# Needs convmerge[tokens] (transformers, no PyTorch).
convmerge tokens -i ./train/decontam.jsonl --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --max-tokens 4096 -o ./train/fit.jsonl

# Train/validation split by content hash: reproducible, order-independent,
# duplicates never straddle the two sides. --val-rows N is exact when N falls between groups;
# it errors rather than splitting duplicate/--keys groups across train and validation.
convmerge split -i ./train/fit.jsonl -o ./train/train.jsonl --val 0.02   # + train.val.jsonl

# Vision-language data: check that every image/audio/video reference exists (local
# paths; --check-urls requests each URL once). -o keeps the rows that pass.
convmerge media -i ./train/vl.jsonl --root ./data -o ./train/vl.ok.jsonl

# Curriculum order: by length, user turns, or a meta score, optionally in stages
# shuffled within (deterministic for --seed). Turn off the trainer's shuffling.
convmerge order -i ./train/train.jsonl -o ./train/train.ordered.jsonl --by length --stages 3

# For trainers that take plain text: {"text": ...} rendered with the model's chat
# template (meta kept). --assistant-spans adds the [start, end) character offsets
# of the assistant turns to train on. Run it last. Needs convmerge[tokens].
convmerge render -i ./train/train.jsonl -o ./train/train.text.jsonl \
  --tokenizer Qwen/Qwen2.5-7B-Instruct --assistant-spans

# LLaMA-Factory: add the dataset_info.json entry for a --format sharegpt file.
convmerge llamafactory-info -i ./data/sft.jsonl --name my_sft --info ./data/dataset_info.json

# axolotl: the datasets: block whose field names match a converted file.
convmerge axolotl-config -i ./train/train.jsonl --val ./train/train.val.jsonl

# Single-turn vs multi-turn report (and split)
convmerge turns -i ./train/train.jsonl --single-out single.jsonl --multi-out multi.jsonl
```

See [docs/quality.md](docs/quality.md) for the `filter` rules, `decontam`, and
near dedupe, [docs/format.md](docs/format.md) for adapter / emitter schemas,
[docs/fetch.md](docs/fetch.md) for manifest details, and
[docs/api.md](docs/api.md) for the Python API and writing plugins
(custom adapters / output formats via entry points).
[docs/stability.md](docs/stability.md) lists what stays compatible across
releases (API, CLI flags and exit codes, file formats).

### 6. `run` — the whole pipeline from one recipe

```yaml
# recipe.yaml
version: 1
output: train/mixed.jsonl
sources:
  alpaca: { path: data/alpaca_data.json, convert: { from: alpaca } }
  tools:
    fetch: { url: https://raw.githubusercontent.com/org/repo/main/tools.jsonl }
    convert: { from: sharegpt }
mix: { total: 100000, seed: 42, weights: { alpaca: 0.7, tools: 0.3 } }
dedupe: true
filter: true                                                         # optional
decontam: { against: [hf:your-org/your-eval-set:main] }              # optional
tokens: { tokenizer: Qwen/Qwen2.5-7B-Instruct, max_tokens: 4096 }   # optional
split: { val: 0.02 }                                                 # optional
```

```bash
convmerge run recipe.yaml --plan     # what would run, and why
convmerge run recipe.yaml            # fetch → normalize → convert → mix → dedupe → filter → decontam → tokens → split
convmerge run recipe.yaml --frozen   # CI: fail unless the lock file is current
convmerge report recipe.yaml --html report.html --card README.md   # read the last run
```

Each step is the same command you would type by hand, so the result is
identical. `recipe.lock.json` records options, convmerge version, and
input/output digests; the next run repeats only the steps whose inputs or
options changed. `convmerge report` turns the last run's `report.json` into
a page you can read (sources and licenses, rows kept and dropped at each
step and why, with examples, turn and answer-length distributions) and a
dataset card for the output. See [docs/recipes.md](docs/recipes.md).

## Out of scope

`convmerge` prepares data for a trainer; it is not one. These stay out:

- **Training models.** No training loop and no wrapper around a trainer.
  convmerge writes files that TRL, LLaMA-Factory, and axolotl read directly,
  plus the settings they need (`llamafactory-info`, `axolotl-config`), and
  checks rows against the target model's chat template (`tokens`).
- **Training-job orchestration**: provisioning GPUs, scheduling and running
  training jobs.
- **Removing personal information (PII).** convmerge does not mask names,
  emails, phone numbers, or IDs. Rules that rewrite training text are easy
  to get wrong without notice (version numbers taken for IP addresses, long
  numbers for card numbers) and still miss names and addresses, so remove
  personal information with a dedicated tool before converting.
  `convmerge filter` with your own `patterns` can find or drop rows that
  match a pattern you choose.

No PyTorch is imported by the core or any extra. The `[tokens]` extra uses
Transformers (without PyTorch) only to render chat templates and count
tokens.

**Not available yet.** These are planned as optional, opt-in steps and are
not part of this release: scoring and labeling rows with a model served over
an HTTP endpoint, building preference pairs, packing, downloading and converting media, and
extracting text from HTML pages. Until then, run them as separate steps of
a larger pipeline and bring the results in as JSON / JSONL / Parquet.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide — setup, local
checks, code conventions, and a walkthrough for adding a new adapter /
emitter. CI runs Ruff, mypy, and pytest on Python 3.10 – 3.14.

```bash
pip install -e ".[dev,all]"
ruff check src tests
ruff format --check src tests
mypy
pytest -q
```

Participation in this project is governed by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

Good first PRs: new adapters / emitters for public dataset schemas, new
fetch backends (GitLab / Zenodo / Kaggle), recipe examples under
[`examples/`](examples/), and docs improvements. Browse the
[`good first issue`](https://github.com/snowmuffin/convmerge/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
label for concrete starting points.

## PyPI release (maintainers)

Releases run from [`.github/workflows/publish.yml`](.github/workflows/publish.yml)
on pushing a `v*` tag. Publishing authenticates via the **`PYPI_API_TOKEN`**
GitHub Actions secret.

1. Create an API token on [pypi.org](https://pypi.org/manage/account/token/).
   - If the project already exists on PyPI, scope the token to the
     `convmerge` project (principle of least privilege).
   - For the very first upload (project not yet registered), PyPI does not
     allow project-scoped tokens — use **Entire account** scope for the
     first release, then rotate to a project-scoped token afterwards and
     revoke the original.
2. In the GitHub repo, *Settings → Secrets and variables → Actions → New
   repository secret*, add `PYPI_API_TOKEN` with the token value.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.

## Changelog

[CHANGELOG.md](CHANGELOG.md) · upgrading: [to 1.0](docs/migration-1.0.md), [from 0.6](docs/migration-0.7.md), [from 0.5](docs/migration-0.6.md)

## License

MIT

### Token checks and raw message fidelity

`tokens` checks native `messages` and conversational preference rows using their
actual stored message dictionaries. In particular, mixed string/object tool-call
arguments are not silently normalized during validation. Other source layouts
are adapted to canonical messages as before. This checks rendering and token
lengths, not the actual trainer's collator or loss mask.

Without `-o`, unreadable/invalid rows or zero measurable rows make the command
exit 1; the JSON report remains available on stdout. With `-o`, it filters rows
and reports what was kept or rejected.
