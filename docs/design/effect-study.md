# Effect study: does convmerge's data preparation change what a model learns?

Status: design fixed before any full run (1.5). Results go to
[docs/effect.md](../effect.md) whatever they show.

## Question

convmerge's checks show that its output loads and trains in TRL, axolotl, and
LLaMA-Factory, and that common hand-written scripts make silent mistakes
(deleted OpenAssistant answers, lost follow-up turns, a mix dominated by long
reasoning rows). This study asks whether those differences show up in the
behaviour of a model trained on the data.

## Setup

**Sources** (streamed from the Hub, first rows of the train split):

| Source | Rows read | What differs between the pipelines |
|---|---|---|
| `OpenAssistant/oasst2` | 20,000 message rows (about 2,000 trees) | tree handling, deleted and rejected replies, follow-up turns |
| `teknium/OpenHermes-2.5` | 5,000 | ShareGPT roles, duplicates |
| `open-r1/OpenR1-Math-220k` | 3,000 | long reasoning rows dominate a row-count mix |
| `tatsu-lab/alpaca` | 5,000 | refusals and empty answers |

**Evaluation set:** `HuggingFaceH4/no_robots`, `test` split (500
human-written conversations), not used for training.

**Pipeline A, common scripts** (`scripts/eval/effect/naive.py`). This is what
dataset cards and tutorials show:

- oasst2: the pandas snippet that pairs each root prompt with its rank-0 reply
  (the same one measured in the 1.3 evaluation);
- OpenHermes: rename `from`/`value` to `role`/`content`, mapping `human` to
  `user` and `gpt` to `assistant`;
- OpenR1-Math: its `messages` column as is;
- alpaca: one user turn (instruction, then the input if any) and one
  assistant turn;
- up to 2,000 rows from each source, concatenated and shuffled.

**Pipeline B, convmerge** (`scripts/eval/effect/recipe.json`): the same raw
files through `convert --from auto`, then these steps:

- `mix` by characters with equal weights, 8,000 rows;
- exact dedupe;
- default `filter`;
- `decontam` against the evaluation set.

**Training (identical for A and B):**

- Model: `HuggingFaceTB/SmolLM2-135M` base weights, with the ChatML chat
  template and `<|im_end|>` as the end token from `SmolLM2-135M-Instruct`.
- Trainer: TRL `SFTTrainer` with its defaults for the loss.
- Schedule: 400 steps at batch size 8, `max_length` 1024, learning rate 3e-4
  with cosine decay and 20 warmup steps, on CPU.
- Seeds: 0, 1, and 2 for each pipeline.

## Metrics (fixed now)

1. **Eval loss**: mean token negative log-likelihood of the assistant turns of
   all 500 evaluation conversations.
2. **Stop rate**: share of 100 greedy generations (first 100 evaluation
   prompts, at most 256 new tokens) that end with `<|im_end|>`.
3. **Role leak**: share of those generations that start another turn:
   `<|im_start|>`, or a line beginning `User:`, `Human:`, `Assistant:`, or
   `### Instruction`.
4. **Refusal**: share containing `as an ai`, `language model`,
   `i'm sorry, but`, `i cannot`, or `i can't assist` (case-insensitive). These
   phrases overlap with convmerge's refusal filter, and that is intended: the
   metric checks whether filtering reaches the model.
5. **Repetition**: share with a word 4-gram that occurs four or more times.
6. **Length**: mean words per generation.

Data statistics are reported as well:

- rows per pipeline and per source;
- each source's share of training tokens;
- what B dropped, and why.

## Reading the results (fixed now)

A metric shows a difference only if two conditions hold:

- all three seeds point the same way;
- the gap between the means is larger than the seed spread (max minus min) of
  both pipelines.

Otherwise it is reported as "no clear difference". Every metric is reported,
including those where A does better.

## What this cannot show

- General capability. A 135M model trained for 400 steps is too small for
  benchmarks to mean anything.
- Anything about larger models or longer training.
- The value of features this setup does not exercise: tool calls, preference
  data, templates other than ChatML.
