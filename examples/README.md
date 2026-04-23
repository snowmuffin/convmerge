# Examples

End-to-end recipe skeletons that exercise the full `convmerge` pipeline.
Each recipe is small, copy-pasteable, and assumes only the installation
from the top-level [README](../README.md):

```bash
pip install "convmerge[all]"
```

The manifests intentionally use `<HF_ORG>/<DATASET>` and `ORG/REPO`
**placeholders** rather than pinning any specific public dataset or
repository. `convmerge` does not endorse or commit to supporting any
particular third-party source — plug in whichever dataset your project
actually uses.

> You are responsible for the license of any data you download. Respect
> each source's terms. `convmerge` does not rehost any dataset.

## Directory layout

- `manifests/` — ready-to-run `convmerge fetch` YAML manifests for the
  common source patterns. Safe defaults (`resume: true`, tokens read
  from env).
- This README walks through the full
  `fetch → normalize → convert → dedupe → turns` sequence for each
  manifest shape.

## Recipes

### Alpaca-style instruction data (HuggingFace)

Any HuggingFace dataset with `instruction` / `input` / `output` columns
works with the built-in `alpaca` adapter.

```bash
# Edit examples/manifests/alpaca_hf.yaml and set `hf:` to your dataset.
convmerge fetch examples/manifests/alpaca_hf.yaml -o ./raw
convmerge normalize -i ./raw/alpaca/train.jsonl -o ./jsonl/alpaca.jsonl
convmerge convert   -i ./jsonl/alpaca.jsonl    -o ./train/alpaca.messages.jsonl \
  --from alpaca --format messages
convmerge dedupe    -i ./train/alpaca.messages.jsonl \
                    -o ./train/alpaca.messages.dedup.jsonl
```

### ShareGPT-style multi-turn (HuggingFace)

Any HuggingFace dataset whose rows look like
`{"conversations": [{"from": ..., "value": ...}, ...]}` works with the
built-in `sharegpt` adapter.

```bash
# Edit examples/manifests/sharegpt_hf.yaml and set `hf:` to your dataset.
convmerge fetch examples/manifests/sharegpt_hf.yaml -o ./raw
convmerge normalize -i ./raw/sharegpt/train.jsonl -o ./jsonl/sharegpt.jsonl
convmerge convert   -i ./jsonl/sharegpt.jsonl    -o ./train/sharegpt.messages.jsonl \
  --from sharegpt --format messages
convmerge turns     -i ./train/sharegpt.messages.jsonl \
  --single-out ./train/sharegpt.single.jsonl \
  --multi-out  ./train/sharegpt.multi.jsonl
```

### Mixed sources, auto-detect (HuggingFace + GitHub)

Combine a HuggingFace dataset with raw JSONL files hosted on GitHub,
then let the heuristic `chat` / `auto` adapter pick the right branch
per record.

```bash
export HF_TOKEN=...     # only needed for gated datasets
export GITHUB_TOKEN=... # only needed for private / rate-limited repos
convmerge fetch examples/manifests/mixed_sources.yaml -o ./raw
convmerge normalize -i ./raw -o ./jsonl
for f in ./jsonl/**/*.jsonl; do
  out="./train/$(basename "$f" .jsonl).messages.jsonl"
  convmerge convert -i "$f" -o "$out" --from auto --format messages
done
convmerge dedupe -i ./train/*.messages.jsonl -o ./train/combined.messages.dedup.jsonl
```

## Contributing a recipe

Recipes are a great documentation contribution. A good recipe:

- Illustrates a **pattern** (e.g. "alpaca-style on HuggingFace",
  "a HF dataset plus raw JSONL on GitHub") rather than committing the
  project to supporting a specific third-party dataset.
- Keeps dataset / repository identifiers as placeholders
  (`<HF_ORG>/<DATASET>`, `ORG/REPO`) unless a concrete name is
  strictly necessary to demonstrate the pattern.
- Is self-contained (one manifest + one shell snippet).
- Does **not** commit the downloaded data — only the manifest and
  walkthrough.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the general contribution
process.
