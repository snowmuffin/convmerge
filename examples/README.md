# Examples

End-to-end recipe skeletons that exercise the full `convmerge` pipeline.
Each recipe is small, copy-pasteable, and assumes only the installation
from the top-level [README](../README.md):

```bash
pip install "convmerge[all]"
```

The manifests in `manifests/` use `<HF_ORG>/<DATASET>` and `ORG/REPO`
**placeholders**; plug in whichever dataset your project uses. The recipes in
`recipes/`, `trl/`, and `llamafactory/` name real public datasets from the
[tested catalog](../README.md#tested-datasets) so they run as written; naming
them is not an endorsement.

> You are responsible for the license of any data you download. Respect
> each source's terms. `convmerge` does not rehost any dataset.

## Directory layout

- `recipes/` — complete `convmerge run` recipes for three common jobs (below).
- `trl/`, `llamafactory/` — the recipes behind the trainer guides in
  [docs/guides](../docs/guides).
- `manifests/` — ready-to-run `convmerge fetch` YAML manifests for the
  common source patterns. Safe defaults (`resume: true`, tokens read
  from env).
- This README walks through the full
  `fetch → normalize → convert → dedupe → turns` sequence for each
  manifest shape.

## Complete recipes

Each one fetches, converts, mixes, cleans, and splits in one command
(`pip install "convmerge[all]"`). Every recipe here is parsed and planned in the
test suite.

| Recipe | Sources | What it shows |
|--------|---------|---------------|
| [recipes/recipe_korean_sft.yaml](recipes/recipe_korean_sft.yaml) | KoAlpaca, KULLM v2, open-korean-instructions | three layouts to `messages`; mix by chars; near dedupe; drop answers that are mostly not Hangul |
| [recipes/recipe_dpo_mix.yaml](recipes/recipe_dpo_mix.yaml) | UltraFeedback, hh-rlhf, PKU-SafeRLHF | scored, transcript, and labeled preference data to `{prompt, chosen, rejected}` |
| [recipes/recipe_agent_tools.yaml](recipes/recipe_agent_tools.yaml) | Glaive, Hermes function calling, smolagents | three tool-call encodings to standard `tool_calls`; rows checked against a Qwen chat template |

```bash
convmerge run examples/recipes/recipe_dpo_mix.yaml --plan   # show the steps
convmerge run examples/recipes/recipe_dpo_mix.yaml          # run them
```

Change `max_rows`, the weights, or `total` to size the output; see
[docs/recipes.md](../docs/recipes.md) for every field.

## Step-by-step walkthroughs

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
