# Examples

End-to-end recipes that exercise the full `convmerge` pipeline against
**public** datasets. Each recipe is small, copy-pasteable, and assumes only
the installation from the top-level [README](../README.md):

```bash
pip install "convmerge[fetch-all,parquet]"
```

> These recipes download data from the public internet. Respect each
> source's license. `convmerge` does not rehost any dataset.

## Directory layout

- `manifests/` — ready-to-run `convmerge fetch` YAML manifests for various
  public sources. Safe defaults (`resume: true`, tokens read from env).
- Per-recipe READMEs below document the full `fetch → normalize → convert
  → dedupe → turns` sequence for a given dataset family.

## Recipes

### Alpaca-style instruction data (HuggingFace)

Pull [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca)
and emit a clean `messages` JSONL.

```bash
convmerge fetch examples/manifests/alpaca_hf.yaml -o ./raw
convmerge normalize -i ./raw/alpaca/train.jsonl -o ./jsonl/alpaca.jsonl
convmerge convert   -i ./jsonl/alpaca.jsonl    -o ./train/alpaca.messages.jsonl \
  --from alpaca --format messages
convmerge dedupe    -i ./train/alpaca.messages.jsonl \
                    -o ./train/alpaca.messages.dedup.jsonl
```

### ShareGPT-style multi-turn (HuggingFace)

Pull
[`anon8231489123/ShareGPT_Vicuna_unfiltered`](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
(or any ShareGPT mirror of your choice) and split single-turn vs multi-turn.

```bash
convmerge fetch examples/manifests/sharegpt_hf.yaml -o ./raw
convmerge normalize -i ./raw/sharegpt/train.jsonl -o ./jsonl/sharegpt.jsonl
convmerge convert   -i ./jsonl/sharegpt.jsonl    -o ./train/sharegpt.messages.jsonl \
  --from sharegpt --format messages
convmerge turns     -i ./train/sharegpt.messages.jsonl \
  --single-out ./train/sharegpt.single.jsonl \
  --multi-out  ./train/sharegpt.multi.jsonl
```

### Mixed sources, auto-detect (HuggingFace + GitHub)

Combine a HuggingFace dataset with raw JSONL files hosted on GitHub, then
let the heuristic `chat` / `auto` adapter pick the right branch per record.

```bash
export HF_TOKEN=...     # only needed for gated datasets
export GITHUB_TOKEN=... # only needed for private / rate-limited repos
convmerge fetch examples/manifests/mixed_sources.yaml -o ./raw
convmerge normalize -i ./raw -o ./jsonl
# One conversion pass per file — use your shell of choice:
for f in ./jsonl/**/*.jsonl; do
  out="./train/$(basename "$f" .jsonl).messages.jsonl"
  convmerge convert -i "$f" -o "$out" --from auto --format messages
done
convmerge dedupe -i ./train/*.messages.jsonl -o ./train/combined.messages.dedup.jsonl
```

## Contributing a recipe

Recipes are a great first contribution. A good recipe:

- Uses a **public** dataset with a clear license.
- Is self-contained (one manifest + one shell snippet).
- States the approximate **download size** and **record count** at the
  top of the README section so readers know what to expect.
- Does **not** commit the downloaded data — only the manifest and
  walkthrough.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the general contribution
process.
