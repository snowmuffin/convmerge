# Fetching datasets with a YAML manifest

The `convmerge fetch` subcommand reads a YAML manifest listing HuggingFace
datasets and/or GitHub URLs and downloads each one into a per-source directory
under a shared output root. It is intentionally a thin layer: HuggingFace
entries delegate to `datasets.load_dataset(...).to_json(...)`, and GitHub
entries use either the stdlib HTTP client, the Trees API, or `git clone`
depending on the entry.

**What it is not:**

- Not a parallel downloader, CDN, or mirror. It calls HuggingFace /
  GitHub directly, one entry at a time, and honours their rate limits.
- Not a replacement for HuggingFace's Arrow cache. Output is a **JSONL
  dump** of the chosen split — convenient for downstream text pipelines,
  but less efficient than `datasets.load_dataset` for repeated random
  access.
- Not a dataset discovery tool. You still name every source explicitly in
  the manifest.

```bash
pip install "convmerge[fetch-all]"   # YAML + HuggingFace datasets (same as [fetch-hf])
# or: pip install "convmerge[all]"    # fetch + parquet + presets
convmerge fetch manifest.yaml -o ./raw
```

## Install extras

| Extra         | Pulls in              | Required for                   |
|---------------|-----------------------|--------------------------------|
| `fetch`       | `pyyaml`              | manifest parsing, GitHub only  |
| `fetch-hf`    | `pyyaml`, `datasets`  | HuggingFace entries            |
| `fetch-all`   | same as `fetch-hf`    | alias; kept for compatibility   |
| `all`         | `pyyaml`, `datasets`, `pyarrow` | full CLI (includes fetch + parquet + presets) |

Raw GitHub URLs and the Trees API use Python's `urllib.request` — no extra
dependency for pure-GitHub manifests beyond PyYAML.

## Manifest schema (version 1)

```yaml
version: 1

defaults:
  output_root: ./raw            # default destination directory
  on_error: continue            # "continue" (default) or "fail"
  resume: true                  # skip entries whose output already exists

auth:
  hf_token_env: HF_TOKEN        # env var to read the HF token from
  hf_token_file: ~/.cache/hf.token
  github_token_env: GITHUB_TOKEN
  github_token_file: ~/.cache/gh.token

datasets:
  # HuggingFace
  - name: alpaca-gpt4-ko
    hf: MarkrAI/KoCommercial-Dataset
    split: train                # optional; defaults to "train"
    config: null                # optional dataset config/subset

  # Single raw file (GitHub raw URL or any direct .json/.jsonl/.json.gz URL)
  - name: orca-math
    url: https://raw.githubusercontent.com/org/repo/main/data/train.jsonl

  # Whole GitHub repo, filtered by extension, via the Trees API (no clone)
  - name: example-repo
    url: https://github.com/org/example-repo
    ext: [".jsonl"]             # only files whose path ends with these

  # Whole GitHub repo, cloned with git (useful for LFS-tracked large files)
  - name: big-lfs-repo
    url: https://github.com/org/big-lfs-repo
    mode: clone
    lfs: true                   # runs ``git lfs pull`` after clone
```

### Required fields per entry

- `name` — unique label used to form the output subdirectory
  (`sanitize_name` strips `<>:"/\|?*` and whitespace).
- Exactly one of `hf` or `url`.
- HuggingFace extras: `split`, `config`.
- GitHub extras: `ext` (tuple of suffixes), `mode` (`tree` default, or `clone`),
  `lfs` (bool, only meaningful when `mode: clone`).
- `output` — optional explicit path that overrides `defaults.output_root / name`.

## Authentication

Tokens are resolved in this order, highest priority first:

1. CLI flags: `--hf-token` / `--github-token`.
2. File at `auth.hf_token_file` / `auth.github_token_file`.
3. Environment variable at `auth.hf_token_env` / `auth.github_token_env`.

Tokens are never printed. Any URL logged by the runner is passed through
`convmerge.fetch.auth.redact_url` to strip `user:token@host` userinfo.

For `mode: clone` entries the token is injected into the clone URL as
`https://user:TOKEN@github.com/...` only when the host is `github.com` or
`huggingface.co`; other hosts receive the URL untouched.

## Resume behaviour

With `defaults.resume: true` (the default) the runner inspects the expected
output path for each entry:

- HuggingFace / raw URL entries: a non-empty file counts as "done".
- Trees / clone entries: a non-empty directory counts as "done".

Pass `--no-resume` on the CLI to force a re-download.

## Running

```bash
# Full manifest
convmerge fetch manifest.yaml -o ./raw

# Only a subset of entries by name
convmerge fetch manifest.yaml --only alpaca-gpt4-ko orca-math

# Fail-fast on any error
convmerge fetch manifest.yaml --on-error fail

# Tokens from CLI (override manifest / env)
convmerge fetch manifest.yaml --hf-token "hf_xxx" --github-token "ghp_xxx"
```

### Single-URL shortcuts (no manifest)

```bash
# HuggingFace dataset -> ./raw/<sanitized>.jsonl
convmerge fetch hf://org/dataset -o ./raw --split train

# Raw GitHub file
convmerge fetch https://raw.githubusercontent.com/o/r/m/a.jsonl -o ./raw

# GitHub repo, Trees API, filtered by extension
convmerge fetch https://github.com/org/repo -o ./raw --ext .jsonl .json

# GitHub repo, full clone with LFS
convmerge fetch https://github.com/org/big-repo -o ./raw --mode clone --lfs
```

## After fetching

`convmerge fetch` only downloads. To actually prepare training data, chain the
other subcommands:

```bash
convmerge fetch    manifest.yaml -o ./raw
convmerge normalize -i ./raw -o ./jsonl         # parquet/json(l) -> clean jsonl
convmerge convert  -i ./jsonl/some.jsonl \
                   -o ./train/some.messages.jsonl \
                   --from auto --format messages
convmerge dedupe   -i ./train/some.messages.jsonl -o ./train/some.dedup.jsonl
convmerge turns    -i ./train/some.dedup.jsonl \
                   --single-out ./train/single.jsonl \
                   --multi-out  ./train/multi.jsonl
```
