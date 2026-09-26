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

Downloads are streamed to disk. When a `raw.githubusercontent.com` URL (or a
file in `mode: tree`) turns out to be a Git LFS pointer, the real object is
fetched through the repository's Git LFS batch API — no clone needed; the
GitHub token (if any) is sent to `github.com` only, never to the object store.
LFS pointers from other hosts are rejected with an actionable error, and the
same error is raised if a pointer reaches normalization.

### Sampling (`max_rows`)

`max_rows: N` on an entry (or `--max-rows N` on the CLI, which applies to every
entry) fetches only the first N records, so `inspect` and trial runs stay
cheap on huge datasets:

| Source | How it samples |
|--------|----------------|
| `hf` | opens the split in `datasets` streaming mode and writes the first N rows; nothing else is downloaded |
| raw URL (`.jsonl` or other line files) | stops downloading after N lines (LFS objects included) |
| `mode: tree` | first N lines of each line file; `.json` files are fetched whole |
| raw `.json` / `.json.gz` | rejected — a JSON array cannot be cut by lines |
| `mode: clone` | not supported (manifest error) |

In streaming mode, HF values JSON cannot represent (e.g. decoded images) are
written as strings.

## Manifest schema (version 1)

```yaml
version: 1

defaults:
  output_root: ./raw            # default destination directory
  on_error: continue            # "continue" (default) or "fail"
  resume: true                  # skip entries with a valid completion marker

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
- `max_rows` — optional positive integer; fetch only the first N records (see
  [Sampling](#sampling-max_rows)).

## Authentication

Tokens are resolved in this order, highest priority first:

1. CLI flags: `--hf-token` / `--github-token`.
2. File at `auth.hf_token_file` / `auth.github_token_file`.
3. Environment variable at `auth.hf_token_env` / `auth.github_token_env`.

Tokens are never printed. Any URL or error logged by the runner is passed
through `convmerge.fetch.auth.redact_url` to strip `user:token@host` userinfo.

Where tokens are sent:

- **Raw URL / Trees API:** the GitHub token is attached only for
  `github.com`, `api.github.com`, and `raw.githubusercontent.com`, and is
  never forwarded when the server redirects. Raw URLs on any other host are
  fetched anonymously.
- **`mode: clone`:** for `github.com` and `huggingface.co` the token is passed
  to `git` (and `git lfs`) as an `Authorization` header scoped to that host
  through git's environment config (`GIT_CONFIG_COUNT`, git ≥ 2.31). It is
  not placed in the clone URL, so it never appears in the process list,
  `.git/config`, or git error messages. Existing clones are pulled with the
  same header, and a token that older convmerge versions (≤ 0.5.0) embedded in
  the `origin` URL is removed on the next fetch. Other hosts are cloned
  without a token.

## Resume behaviour

With `defaults.resume: true` (the default), a successful fetch writes a
`<output>.fetch.json` sidecar containing a versioned file or directory
snapshot. The runner skips an entry only when that marker still matches the
output. This prevents a truncated or manually modified output from being
silently treated as complete. Existing non-empty outputs without a marker are
fetched once to establish the completion record.

The output snapshot uses the file size and SHA-256 digest for files. Directory
outputs record the relative files, sizes, and digests. A marker write failure
does not discard a successful download; the next resume conservatively fetches
it again.

A sampled fetch records its `max_rows` in the marker, so a sample never
satisfies a later full fetch (or a sample of a different size).

With the marker present and valid, the runner skips:

- HuggingFace / raw URL entries: the downloaded file.
- Trees / clone entries: the complete directory snapshot.

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
