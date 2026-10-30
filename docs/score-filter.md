# Score-filter study: results

With the same number of training tokens, does a small model trained on the
best-rated rows of a dataset do better than one trained on random rows of the
same dataset? The design, metrics, reading rule and gate decision were fixed
before the run: [design/score-filter-study.md](design/score-filter-study.md).
This page reports every result, including those where the random rows did
better.

**Short answer: in this setup, no.** The model trained on the best-rated rows
had the same eval loss as the one trained on random rows, within the spread of
the seeds, and no behaviour metric differed. Gate A therefore **fails**: the
thin model-features stage of the improvement plan is reduced to labeling, and
score-based filtering is not promoted as a quality feature.

## Setup in one paragraph

The pool is the HelpSteer2 train split, whose responses carry human ratings
from 0 to 4. It was prepared once for both arms:

1. `convert --keep-fields helpfulness,correctness`, which carries the ratings
   into `meta`;
2. `tokens --max-tokens 512 -o`, which drops rows longer than the training
   window;
3. `decontam` against the evaluation set.

Two arms were taken from that pool:

- **S:** rows sorted by helpfulness, then correctness, both descending.
- **R:** the same rows shuffled with seed 0.

Each arm took rows until it held the token budget plus 5%. Both arms then
went through the same training:

- **Model:** SmolLM2-135M base weights with the ChatML template.
- **Trainer:** TRL `SFTTrainer`, its default loss, packed into full 512-token
  sequences.
- **Schedule:** 400 steps × 8 sequences × 512 tokens = 1,638,400 training
  tokens per run.
- **Seeds:** 0 to 4 per arm, on CPU.

Evaluation used the 500 human-written conversations of `no_robots` (test
split), without packing, plus 100 greedy generations per run.

Run: [Actions run 41075288502](https://github.com/snowmuffin/convmerge/actions/runs/41075288502),
convmerge at commit `6f4321c`. The per-run metrics are in that run's
`score-filter-result-*` artifacts.

## Results

A metric counts as a difference only if two things hold:

- all five seeds point the same way;
- the gap between the means is larger than the seed spread (max − min) of
  both arms.

| Metric | R, random rows: mean (min–max) | S, best-rated rows: mean (min–max) | Reading |
|---|---|---|---|
| Eval loss, assistant tokens (lower is better) | 2.601 (2.598–2.603) | 2.605 (2.601–2.608) | no clear difference |
| Stops on its own (higher is better) | 0.354 (0.28–0.40) | 0.346 (0.28–0.41) | no clear difference |
| Starts another turn (lower is better) | 0.000 | 0.000 | no clear difference |
| Refusal phrases (lower is better) | 0.044 (0.00–0.10) | 0.020 (0.01–0.03) | no clear difference |
| Repeated word 4-gram ×4 (lower is better) | 0.652 (0.62–0.67) | 0.680 (0.65–0.71) | no clear difference |
| Mean words per answer | 160.9 (155.8–167.2) | 163.7 (158.1–167.5) | no clear difference |
| Train loss (different data, not comparable) | 1.638 (1.635–1.641) | 1.617 (1.615–1.618) | S lower |

**Gate A: fail.** The primary metric, eval loss, shows no clear difference.
The pre-registered rule counts anything other than "S better" on eval loss as
a fail.

## Training data

| | Pool | R | S |
|---|---|---|---|
| Rows | 11,646 | 5,461 | 5,444 |
| Tokens (training template) | 3,655,258 | 1,720,336 | 1,720,408 |
| Mean response tokens | 222.5 | 223.0 | 231.4 |
| Distinct prompts | 6,917 | 4,453 | 4,454 |
| Helpfulness 0 / 1 / 2 / 3 / 4 | 993 / 992 / 1,275 / 3,323 / 5,063 | 468 / 465 / 569 / 1,528 / 2,431 | 0 / 0 / 0 / 381 / 5,063 |
| Correctness 0 / 1 / 2 / 3 / 4 | 918 / 988 / 1,093 / 2,848 / 5,799 | 435 / 456 / 500 / 1,292 / 2,778 | 0 / 9 / 14 / 40 / 5,381 |

- **Where S stops.** S took every helpfulness-4 row and then helpfulness-3
  rows down to correctness 4. 27.5% of R's rows (1,502) are rated 0–2 for
  helpfulness, and S has none.
- **Overlap.** Each arm takes about 47% of the pool, so the arms share 2,609
  rows (48% of each) and 3,021 prompts.
- **Token budget.** The two arms hold the same number of tokens to within
  0.005%. Packing fills every sequence, so every run trained on
  400 × 8 × 512 tokens. Token count
  does not explain the result, unlike in the [effect study](effect.md).

## What the numbers say

1. **Leaving out the low-rated quarter changed nothing measurable.** R trained
   on 1,502 rows that people had rated unhelpful, and S on none. The eval
   losses differ by 0.004, R's being the lower, and that gap is inside the
   seed spread of both arms.
2. **The best-rated rows are easier to fit, and that did not carry over.** S
   reached a lower train loss in every seed. Its rows are more alike, mostly
   helpfulness 4 and correctness 4. The model learned them better but did
   not do better on held-out human-written conversations.
3. **Behaviour did not move either.** None of the generation metrics differs
   under the rule. S's refusal rate is lower on average (0.020 against
   0.044), but R's seeds range from 0.00 to 0.10, so the seeds do not agree.

## What this cannot show

The [design](design/score-filter-study.md#what-this-cannot-show) lists what
the study cannot answer. In particular, it says nothing about the following:

- scores from a model judge;
- larger models or longer training;
- preference (DPO) data;
- a pool where the low-rated rows are worse than HelpSteer2's.

HelpSteer2's low ratings mark weaker answers to real prompts, not broken
rows. Broken rows (empty answers, refusals, loops) are what `filter` already
drops by rule.

The arms share half their rows, which narrows the difference between them. A
study with disjoint arms would need a pool more than twice the budget after
preparation, and this pool is 2.1 times the budget.

## What changes in convmerge

The design fixed this decision before the run:

- **The thin model-features stage is reduced to labeling.** Scores and
  labels from a model may be written to `meta` for mixing, ordering and
  reports. convmerge does not ship score-based filtering as a quality step.
- **`filter --meta-min` stays.** It is a general rule for scores you already
  have. The docs now say that filtering by human helpfulness ratings did not
  help in this setup ([quality.md](quality.md#scores-under-meta---meta-min)).

## Run notes

- **Step time.** The dry run ([run 41073050576](https://github.com/snowmuffin/convmerge/actions/runs/41073050576),
  20 steps) measured 22.1 s per step. The full run, ten jobs at once,
  averaged 34 s per step (28.5–36.7). Each job took between 3 h 21 min and
  4 h 20 min, inside the 340-minute limit.
- **No amendments after Amendment 1.** The packing strategy amendment was
  written before any run. Nothing in the design changed after the dry run.
