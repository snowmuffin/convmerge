# Effect study: results

Does preparing data with convmerge change what a small model learns, compared
with the conversion scripts people usually write? The design, metrics, and
reading rule were fixed before the runs:
[design/effect-study.md](design/effect-study.md). This page reports every
result, including those where the common scripts did better.

**Short answer: in this setup, no.** The model trained on convmerge's output was
no better on any behaviour metric. It was slightly worse on eval loss, and the
likely reason is that it saw fewer training tokens (below). convmerge's value
shows in the data, where it is measurable. This study found no sign of it in a
small model's behaviour.

## Setup in one paragraph

Four public sources (oasst2, OpenHermes-2.5, OpenR1-Math, alpaca) were prepared
in two ways:

- **A:** the scripts shown on dataset cards and in tutorials, 2,000 rows per
  source;
- **B:** one convmerge recipe: `convert --from auto`, mix by characters with
  equal weights, exact dedupe, default `filter`, and `decontam` against the
  evaluation set.

Both pipelines then went through the same training and evaluation:

- **Model:** SmolLM2-135M (base weights) with the ChatML template.
- **Training:** TRL `SFTTrainer`, 400 steps of 8 rows, `max_length` 512,
  3 seeds per pipeline, CPU.
- **Evaluation:** 500 human-written conversations from `no_robots` (test
  split), and 100 greedy generations per run.

Run: [Actions run 36748877506](https://github.com/snowmuffin/convmerge/actions/runs/36748877506),
convmerge at commit `c1e4c13`.

## Results

A metric counts as a difference only if all three seeds point the same way
**and** the gap between the means is larger than the seed spread of both
pipelines.

| Metric | A, common scripts: mean (min–max) | B, convmerge: mean (min–max) | Reading |
|---|---|---|---|
| Eval loss, assistant tokens (lower is better) | 2.603 (2.601–2.605) | 2.612 (2.608–2.615) | **A better** |
| Stops on its own (higher is better) | 0.497 (0.47–0.53) | 0.540 (0.51–0.56) | no clear difference |
| Starts another turn (lower is better) | 0.000 | 0.000 | no clear difference |
| Refusal phrases (lower is better) | 0.023 (0.00–0.04) | 0.007 (0.00–0.02) | no clear difference |
| Repeated word 4-gram ×4 (lower is better) | 0.533 (0.50–0.56) | 0.520 (0.50–0.55) | no clear difference |
| Mean words per answer | 116.0 (112.5–117.8) | 112.4 (110.8–113.6) | no clear difference |
| Train loss (different data, not comparable) | 1.574 (1.562–1.584) | 1.677 (1.674–1.680) | — |

Per run:

| Run | Eval loss | Stop | Leak | Refusal | Repeat | Words | Tokens trained |
|---|---|---|---|---|---|---|---|
| A seed 0 | 2.6008 | 0.47 | 0 | 0.00 | 0.56 | 117.7 | 1,003,000 |
| A seed 1 | 2.6052 | 0.53 | 0 | 0.03 | 0.54 | 117.8 | 1,009,000 |
| A seed 2 | 2.6019 | 0.49 | 0 | 0.04 | 0.50 | 112.5 | 1,012,000 |
| B seed 0 | 2.6084 | 0.51 | 0 | 0.00 | 0.50 | 112.7 | 604,400 |
| B seed 1 | 2.6131 | 0.56 | 0 | 0.00 | 0.55 | 113.6 | 607,500 |
| B seed 2 | 2.6149 | 0.55 | 0 | 0.02 | 0.51 | 110.8 | 595,400 |

"Tokens trained" is the trainer's own count after truncation to 512.

## Training data

| Source | A rows | A token share (full rows) | B rows | B token share (full rows) |
|---|---|---|---|---|
| OpenR1-Math | 2,000 | 88.3% | 112 | 30.4% |
| oasst2 | 2,000 | 5.7% | 1,194 | 32.1% |
| OpenHermes-2.5 | 2,000 | 5.0% | 1,234 | 21.4% |
| alpaca | 2,000 | 1.0% | 4,996 | 16.1% |
| **Total** | **8,000** | | **7,536** | |

## What the numbers say

1. **A's lower eval loss is most likely a token-count effect.** Both pipelines
   ran 400 steps of 8 rows, but A's steps held 1.66 times as many tokens:
   - In A, 2,000 OpenR1 rows of about 6,400 tokens each were each cut to 512,
     so a quarter of A's rows fill the whole window.
   - In B, the character-weighted mix took almost every alpaca row to reach
     alpaca's share. Alpaca rows are short, so two-thirds of B's rows are
     short.

   The study was not designed to separate data quality from token count, so
   this cannot be settled here. The gap itself is small, 0.35%.

2. **The mix weights were set on text the trainer never saw.** B's mix
   balanced the characters of whole rows. The trainer then cut every row to
   512 tokens, which removed most of OpenR1's share in both pipelines. In A
   this made the scripts' worst problem (88% of tokens from one source) mostly
   disappear. In B it turned an even text mix into a mix dominated by short
   alpaca rows. The lesson: when training at a fixed length, the weights
   should count each row only up to that length. Since 1.6,
   `mix --by tokens --max-tokens N` (recipe `mix.max_tokens`) does this and
   reports each source's share of rows longer than N. Before 1.6 the way was
   to drop the longer rows from each source first
   (`convmerge tokens --max-tokens N -o`) and mix the results.

3. **No role leaks in either pipeline.** The common scripts used here map
   roles correctly: OpenHermes `human`/`gpt` to `user`/`assistant`. The
   failure convmerge's role mapping prevents did not occur, so there was
   nothing for it to fix.

4. **Refusal filtering did not reach the model measurably.** B filtered
   refusals, and B's refusal rate was lower on average (0.7% against 2.3%).
   The seeds overlap, so under the pre-set rule this is not a difference.

5. **Behaviour is dominated by model size.** About half the generations of
   both pipelines never stop within 256 tokens and repeat a 4-gram. A 135M
   model trained for 400 steps is too small to show finer differences.

## What this does not show

- Anything about larger models, longer training, or training without
  truncation.
- The parts of convmerge this setup does not use: tool calls, preference
  data, templates other than ChatML, the chat-template checks in `tokens`.
- Time saved. convmerge replaced four scripts with one recipe. That is its
  practical value, and this study does not measure it.

## Reproducing

The workflow is `.github/workflows/effect.yml`. Once on the default branch it
runs by hand (Actions → Effect study → Run workflow) with the defaults above.
Six training jobs of about 2 to 3.5 hours each run in parallel on standard
runners. The scripts are in `scripts/eval/effect/`.
