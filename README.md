# convmerge

Fetch, normalize, and convert heterogeneous chat / instruct datasets into a
**single LLM training format** (JSONL).

**Repository:** [github.com/snowmuffin/convmerge](https://github.com/snowmuffin/convmerge)  
**Status:** pre-1.0; APIs and CLI may change between minor versions.

## Install

```bash
pip install convmerge                    # core: convert, normalize, dedupe, turns
pip install "convmerge[fetch]"           # + YAML manifest fetcher (GitHub)
pip install "convmerge[fetch-hf]"        # + HuggingFace entries (adds ``datasets``)
pip install "convmerge[fetch-all]"       # all fetch-related extras
pip install "convmerge[parquet]"         # + parquet streaming input
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
```

Adapters: `alpaca`, `sharegpt`, `chat` (alias `auto`).  
Emitters: `messages`, `alpaca`.

### 4. `dedupe` / `turns` — final cleanup + train/eval split hook

```bash
convmerge dedupe -i ./train/mixed.messages.jsonl -o ./train/mixed.dedup.jsonl
convmerge turns  -i ./train/mixed.dedup.jsonl \
  --single-out ./train/single.jsonl \
  --multi-out  ./train/multi.jsonl
```

See [docs/format.md](docs/format.md) for adapter / emitter schemas and
[docs/fetch.md](docs/fetch.md) for manifest details.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). CI runs Ruff + pytest on Python
3.10 – 3.12.

```bash
ruff check src tests
pytest -q
```

## PyPI release (maintainers)

Releases run from [`.github/workflows/publish.yml`](.github/workflows/publish.yml)
on pushing a `v*` tag. Publishing authenticates via the **`PYPI_API_TOKEN`**
GitHub Actions secret (a PyPI API token scoped to the `convmerge` project).

1. Create an API token on [pypi.org](https://pypi.org/manage/account/token/)
   scoped to `convmerge`.
2. In the GitHub repo, *Settings → Secrets and variables → Actions → New
   repository secret*, add `PYPI_API_TOKEN` with the token value.
3. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`.

## Changelog

[CHANGELOG.md](CHANGELOG.md)

## License

MIT
