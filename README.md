# convmerge

[![PyPI](https://img.shields.io/pypi/v/convmerge.svg)](https://pypi.org/project/convmerge/)
[![Python versions](https://img.shields.io/pypi/pyversions/convmerge.svg)](https://pypi.org/project/convmerge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/snowmuffin/convmerge/actions/workflows/ci.yml/badge.svg)](https://github.com/snowmuffin/convmerge/actions/workflows/ci.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/convmerge.svg)](https://pypi.org/project/convmerge/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

`convmerge` is a **data-preparation library** for supervised fine-tuning (SFT)
datasets. It fetches, normalizes, and merges heterogeneous chat / instruct
sources into a single newline-delimited JSON Lines layout that training code
can consume directly.

It is intentionally scoped to the **pre-training-loop** step: no model
loading, no inference, no labeling, no training orchestration. See
[Out of scope](#out-of-scope) below.

**Repository:** [github.com/snowmuffin/convmerge](https://github.com/snowmuffin/convmerge)  
**Status:** pre-1.0; APIs and CLI may change between minor versions until 1.0.

## Install

```bash
pip install convmerge                    # core: convert, normalize, dedupe, turns
pip install "convmerge[fetch]"           # + YAML manifest fetcher (GitHub)
pip install "convmerge[fetch-hf]"        # + HuggingFace entries (adds ``datasets``)
pip install "convmerge[fetch-all]"       # all fetch-related extras
pip install "convmerge[parquet]"         # + parquet streaming input
pip install "convmerge[preset]"           # + YAML convert presets (--preset, preset validate)
```

Or from a clone:

```bash
git clone https://github.com/snowmuffin/convmerge.git
cd convmerge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,fetch-all,parquet]"
```

## The four use cases

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
  - { name: alpaca-ko, hf: MarkrAI/KoCommercial-Dataset, split: train }
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
single-line JSON (`{...}{...}{...}`), and already-valid JSONL. A directory
input is walked recursively and mirrored under the output directory.

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

Adapters: `alpaca`, `sharegpt`, `chat` (alias `auto`).  
Emitters: `messages`, `alpaca`.

Presets and team-specific tuning: [docs/custom_presets.md](docs/custom_presets.md).

> `chat` / `auto` is a **heuristic** adapter: it inspects the keys of each
> input record (`messages`, `conversation(s)`, `text`, `conversation_a`/`_b`,
> `instruction`/`input`/`output`, …) and routes to the right branch with a
> configurable role map. For unusual schemas, pin an explicit adapter
> (`alpaca`, `sharegpt`) or override keys programmatically — see
> [docs/format.md](docs/format.md).

### 4. `dedupe` / `turns` — final cleanup + train/eval split hook

```bash
convmerge dedupe -i ./train/mixed.messages.jsonl -o ./train/mixed.dedup.jsonl
convmerge turns  -i ./train/mixed.dedup.jsonl \
  --single-out ./train/single.jsonl \
  --multi-out  ./train/multi.jsonl
```

See [docs/format.md](docs/format.md) for adapter / emitter schemas and
[docs/fetch.md](docs/fetch.md) for manifest details.

## Out of scope

To keep the package lean and dependency-free at its core, `convmerge` does
**not** include — and has no plans to include — the following:

- **Model loading / inference / training.** No PyTorch, Transformers, vLLM,
  or similar runtime is imported by the core or any shipped extra.
- **Automatic labeling or classification of samples** (e.g. topic tagging,
  quality scoring, safety classification). These are left to upstream tools
  or private pipelines.
- **RLHF / DPO / preference-dataset construction** beyond passing through
  existing pairwise rows via the `chat` adapter's `pairwise_mode`.
- **Training-job orchestration** (SkyPilot, RunPod, Modal, K8s operators).
- **Prompt templating / chat-template rendering** for specific model
  families. Output JSONL uses the standard `messages` / `alpaca` shapes;
  downstream trainers apply their own template.
- **Tokenizer-aware length filtering, packing, or curriculum scheduling.**
  Those live in the training stack, not here.
- **Scraping HTML pages or running browser automation.** Structured JSON /
  JSONL / Parquet inputs only.

If any of these are important to your workflow, wire `convmerge` in as one
step of a larger pipeline rather than expecting it to grow into those areas.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide — setup, local
checks, code conventions, and a walkthrough for adding a new adapter /
emitter. CI runs Ruff + pytest on Python 3.10 – 3.12.

```bash
pip install -e ".[dev,fetch-all,parquet]"
ruff check src tests
ruff format --check src tests
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

[CHANGELOG.md](CHANGELOG.md)

## License

MIT
