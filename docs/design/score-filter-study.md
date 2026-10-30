# Score-filter study: does keeping the best-scored rows train a better model?

Status: **draft design**. It becomes fixed when merged to `main`, before any
full run; that commit is the pre-registration. Changes after that go in an
"Amendment" section, written before any result is seen, as in the
[effect study](effect-study.md).

This is gate A of the improvement plan: convmerge builds model-based scoring
and labeling (the "thin model features" stage) only if this study passes.

## Question

With the same number of training tokens, does a small model trained on the
highest-scored rows of a dataset do better than one trained on rows picked at
random from the same dataset?

The scores here are human ratings, not a model's. The study tests the premise
behind score-based filtering: that keeping well-rated rows helps. It does not
test a particular scoring model.

## Data

**Pool:** `nvidia/HelpSteer2`, train split. Each row is a prompt and one
response with human ratings from 0 to 4 for helpfulness, correctness,
coherence, complexity and verbosity. The license and terms on its dataset card
are checked before the run.

Preparation, the same for both arms:

1. `convert` to `messages`, carrying `helpfulness` and `correctness` into
   `meta`.
2. Drop rows longer than 512 tokens under the training chat template
   (`convmerge tokens --max-tokens 512 -o`), so that no row is cut during
   training.
3. `decontam` against the evaluation set.

**Arms**, taken from that one pool:

- **S (scored):** rows ordered by `helpfulness`, then `correctness`, both
  descending. Ties are ordered by a hash of the row.
- **R (random):** the same rows shuffled with a fixed seed (0). This seed is
  independent of the training seeds.

Each arm takes rows in its order until it holds at least the token budget
below plus 5%. Rows are counted under the training template. The score cut
that S ends at is reported.

## Training (identical for both arms)

- Model: `HuggingFaceTB/SmolLM2-135M` base weights, ChatML template and
  `<|im_end|>` end token from `SmolLM2-135M-Instruct`, as in the effect study.
- Trainer: TRL `SFTTrainer`, its default loss, **with packing**. Every step
  then trains exactly 8 sequences of 512 tokens, so both arms see the same
  token count: 400 × 8 × 512 = 1,638,400 tokens.
- Schedule: 400 steps, batch 8 (4 per step, 2 accumulated), learning rate
  3e-4, cosine decay, 20 warmup steps, CPU on a standard GitHub runner.
- Seeds: 0, 1, 2, 3 and 4 per arm, so 10 runs.

## Metrics (fixed now)

**Primary:** eval loss, the mean token negative log-likelihood of the
assistant turns of `HuggingFaceH4/no_robots` (test split, 500 human-written
conversations). It is computed exactly as in the effect study, without
packing.

**Secondary** (reported, never decide the gate): over 100 greedy generations,
the stop rate, role leak, refusal phrases, repetition and mean words, with
the definitions in the [effect study](effect-study.md).

**Data statistics:** for each arm, report the following.

- rows, training tokens, and mean response length
- the helpfulness and correctness distributions
- what preparation dropped, and why

## Reading the results (fixed now)

A metric shows a difference only if both conditions hold:

- all five seeds point the same way;
- the gap between the means is larger than the seed spread (max minus min) of
  both arms.

Otherwise it is "no clear difference". Every metric is reported, including
those where R does better.

## Gate decision (fixed now)

- **Pass:** S has a lower eval loss than R under the reading rule above. The
  thin model features stage goes ahead as planned.
- **Fail:** anything else, which means no clear difference or R better. That
  stage is reduced to labeling: labels are written to `meta` for mixing and
  reports. Score-based filtering is not promoted as a quality feature, and
  the docs state that it did not help in this setup.

## Known confounds

- **Length.** Better-rated responses may be longer, so S may hold fewer, longer
  rows than R. Packing equalises tokens, not rows, so row counts and lengths
  are reported next to the result.
- **Packing.** Several conversations share one 512-token sequence. Whether
  they can attend to each other depends on the attention implementation used
  on CPU. Both arms are affected alike, and the eval loss is computed without
  packing.
- **Prompt overlap.** HelpSteer2 has about two responses per prompt, so a
  prompt can appear in both arms, or twice in one arm.
- **Model size.** A 135M model trained on 1.6M tokens is small. A pass shows
  that the data effect reaches the model. A fail does not show that it never
  would at larger scale.

## What this cannot show

- Anything about scores from a model judge, larger models or longer training.
- Whether filtering helps for preference (DPO) data.

## Before the run (not part of the fixed design)

1. Implement the plan's first-stage item: carry score columns into `meta` at
   convert time and filter by them. The study runs on that release.
2. Add a preparation script that builds the pool and both arms, and add
   `--packing` to `scripts/eval/effect/train_eval.py`. Packing is off by
   default, so the effect study stays reproducible.
3. Add a workflow, or inputs to `.github/workflows/effect.yml`, for the two
   arms and five seeds.
4. Do a dry run at a small scale and record the step time. In the effect
   study a step at 1,024 tokens took about 34 s on a standard runner, so 400
   steps at 512 should take about 2 hours. That is an estimate, and the dry
   run checks it against the 6-hour job limit. If the setup has to change to
   fit, the change is recorded as an amendment before any full run.
