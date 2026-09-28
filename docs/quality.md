# Quality checks: `filter`, `decontam`, `dedupe --near`

Three commands remove rows that should not reach training. All three are
deterministic and CPU-only, and they call no model. Each one:

- reads any layout `convert` reads (`messages`, ShareGPT, Alpaca, preference
  pairs, …);
- writes the rows it keeps byte-for-byte as read (`-o`), and the rows it
  drops to `--rejects` if you ask for them;
- prints a JSON report (`"version": 1`) to stdout, with example rows for each
  finding.

Without `-o`, `filter` and `decontam` only report, so you can look at what
would go before dropping anything.

| Command | Drops | Needs |
|---|---|---|
| `filter` | Refusals, empty answers, looping text, broken preference pairs; optionally slop, length, script, and your own regexes | core |
| `decontam` | Rows that share a 13-word n-gram with an evaluation set | core (`hf:` sets: `[fetch-all]`) |
| `dedupe --near` | Near-copies of an earlier row | `[quality]` (datasketch) |

In a recipe they run after `dedupe` and before `tokens`
([recipes.md](recipes.md#filter-and-decontam)).

## `convmerge filter`

```bash
convmerge filter -i mixed.jsonl                                   # report only
convmerge filter -i mixed.jsonl -o clean.jsonl --rejects rejected.jsonl
convmerge filter -i ko.jsonl -o clean.jsonl --enable slop --min-script hangul=0.3
```

### Rules

| Rule | Default | A row is dropped when |
|---|---|---|
| `empty_answer` | on | Its final assistant answer is empty or shorter than `--min-answer-chars` (default 1). An answer that is a tool call is not empty. |
| `refusal` | on | An answer refuses, or carries an "as an AI language model" disclaimer. The list covers English and Korean. Disclaimers count anywhere in the answer. Refusal openings ("I'm sorry, but I can't…", "죄송하지만…", "도와드릴 수 없습니다") count only when they start the answer or a sentence within its first 200 characters, so an answer that *talks about* declining is kept. In a conversation with a system prompt or tools, only disclaimers count: an assistant scoped to a task (a tool-calling model, a support persona) declines out-of-scope requests on purpose, and that is worth training. |
| `repetition` | on | An answer or a reasoning trace loops: at least `--repetition-max` (default 0.7) of its word 10-grams repeat an earlier 10-gram. Texts under 50 words are not checked. Real loops measured 75–100%; song choruses, repeated prompt templates, and SCAN-style action lists measured 53–65%. |
| `near_identical_pair` | on | (Pairs) The chosen and rejected answers are the same once case and whitespace are ignored. |
| `rejected_empty` | on | (Pairs) The rejected answer is empty. |
| `length` | off | The answer text is shorter than `--min-chars` or longer than `--max-chars`. Either flag turns the rule on. |
| `slop` | off | The answers use `--slop-max` (default 3) or more stock phrases, such as "delve into", "it's important to note that", "a testament to", or "도움이 되셨길 바랍니다". |
| `script` | off | Less than the given share of the conversation's letters are in a script. For example, `--min-script hangul=0.3` catches half-translated rows. Code blocks and inline code are ignored. Scripts: `hangul`, `latin`, `han`, `kana`, `cyrillic`. Texts under 20 letters are not checked. |

`--enable RULE` and `--disable RULE` switch rules on and off, and both can
be repeated.

For a preference pair, the rules read the **chosen** answer only. A refused
or looping *rejected* answer is what DPO learns to avoid, so it is kept.

### Your own phrases and patterns

```yaml
# rules.yaml  (--rules-file rules.yaml; JSON works too)
refusal: ["I would rather not"]       # added to the built-in refusal phrases (matched anywhere)
slop: ["synergy", "game-changer"]     # added to the slop phrases
builtin_phrases: true                 # false: use only the phrases above
patterns:                             # each pattern is a rule of its own
  url: "https?://"
  placeholder: "\\[(?:INSERT|TODO)[^\\]]*\\]"
```

Phrases are matched case-insensitively, with typographic quotes treated as
plain ones. Patterns are Python regular expressions, searched in each
answer.

### Report

```json
{
  "version": 1,
  "rows": 20000, "kept": 19412, "rejected": 588,
  "unreadable": 0, "invalid_json": 0,
  "rules": {"empty_answer": 12, "refusal": 431, "repetition": 45, "...": 0},
  "samples": {"refusal": [{"line": 118, "text": "I'm sorry, but I can't help with..."}]},
  "preference": {"pairs": 0, "...": 0}
}
```

- `rules` counts the rows each rule matched. A row can match several rules,
  and `rejected` counts it once.
- `samples` holds up to three examples per rule; `repetition` examples start
  with the share of repeated n-grams they measured. Read them before you trust
  a rule on a new dataset.
- `unreadable` counts rows in no layout convmerge knows. They are dropped.
- `preference` appears when the file has pairs. It holds `pairs`,
  `chosen_longer` (pairs whose chosen answer is the longer one),
  `chosen_longer_share`, and `median_length_ratio` (chosen length divided by
  rejected length). When more than 70% of at least 20 pairs have the longer
  answer as the chosen one, `filter` warns: DPO on such data tends to learn
  "longer is better".

## `convmerge decontam`

```bash
convmerge decontam -i train.jsonl --against hf:openai/gsm8k:main \
  --against hf:cais/mmlu:all --against evals/internal.jsonl -o clean.jsonl
```

- `--against` takes a JSONL file or `hf:REPO[:CONFIG[:SPLIT]]`, and can be
  repeated. The split defaults to `test`, and `hf:` needs
  `convmerge[fetch-all]`.
- Each evaluation row becomes **one passage**: its string fields in order
  (lists included). So a multiple-choice question and its options form one
  text, and a training row that copies the question with its options
  matches. `--fields question,choices` limits which fields are read.
- Text is compared as lower-cased words without punctuation.
  - Han and kana characters count as one word each, so Chinese and Japanese
    work without spaces.
  - Korean is compared by its space-separated words.
  - Single Latin letters and single digits are ignored on both sides, since
    they are mostly option labels ("A.", "(b)", "1)").
- A training row is dropped when it contains any 13-word run of a passage
  (`--ngram`).
  - A passage shorter than that must appear whole.
  - A passage shorter than `--min-tokens` (default 8) is skipped, because
    it is too generic to match on. The report counts these as `too_short`.
- `--check prompts` (the default) looks at system and user turns and a
  pair's prompt. `--check all` also looks at answers, which catches copied
  solutions.

Rows in no layout convmerge reads are kept (there is nothing to compare)
and counted as `unreadable`.

The report lists every evaluation set with the rows it read, the passages
it indexed, the passages too short to index, and the training rows it
`matched`. It also shows up to five examples, each naming the evaluation
set that matched.

Word n-grams catch copies, including lightly edited ones. They do not catch
paraphrases or translations of a benchmark: a Korean translation of GSM8K
shares no words with GSM8K.

## `convmerge dedupe --near`

```bash
pip install "convmerge[quality]"
convmerge dedupe -i mixed.jsonl -o deduped.jsonl --near            # threshold 0.8
convmerge dedupe -i mixed.jsonl -o deduped.jsonl --near --threshold 0.7
```

- Each row's text is every turn of the conversation except the system
  prompt, or both sides of a pair; with `--keys`, the values of those keys
  instead. The text is cut into word 5-grams. System prompts are left out
  because they are often shared templates (tool definitions, personas) that
  would make unrelated rows look alike.
- Rows whose estimated Jaccard similarity to an earlier kept row reaches
  `--threshold` are dropped, so the first row of a group of near-copies
  stays.
- This catches what exact `dedupe` misses: the same source translated or
  reformatted slightly differently, or answers that differ in a few words.
- It is approximate. MinHash estimates the similarity, and LSH finds the
  candidates.
- The index stays in memory. On 200,000 chat rows (4 KB each) the peak
  was 733 MB with the default `--num-perm 128` and 445 MB with
  `--num-perm 64`, which removed nearly the same rows (6.78% and 6.70%).
  That is about 3.5 KB and 2 KB per row, or roughly 3.5 GB and 2 GB per
  million rows; it ran at about 2,300 rows per second either way. Beyond a
  few million rows use a distributed tool such as
  [datatrove](https://github.com/huggingface/datatrove).
- `--threshold` must be below 1. Very high thresholds (0.95 and up) need a
  larger `--num-perm`.

## How the defaults were chosen

The rules are measured on real Hub rows. `scripts/quality.py` streams the
first rows of every dataset in the [tested-datasets
catalog](../README.md#tested-datasets) and runs every rule, `decontam`
against GSM8K, MMLU, KMMLU, and KoBEST, and `dedupe --near`. It lists
example rows per rule, which are the data for judging false positives. The
**Quality** GitHub workflow runs it on pull requests that change the rules.

The measurement for 0.14 used the first 1,000 rows of each of 43 catalog
datasets:

- **`refusal`:** It matched 0–3% of rows. Nearly all matches were "as an AI
  language model" disclaimers (guanaco, HelpSteer2, UltraFeedback, sharegpt-korean)
  or plain refusals (KULLM v2 "죄송하지만 …"). Tool-calling and
  system-prompted data (glaive, Multilingual-Thinking) first matched 2–3%
  on purposeful out-of-scope declines. Since then, only disclaimers count
  in those conversations, and they match 0%.
- **`repetition`:** Loops measured 75–100% repeated, for example a
  translation stuck on one phrase (16 of 1,000 rows of
  ko_Ultrafeedback_binarized) and a response pasting the same prompt five
  times (rated helpfulness 0 in HelpSteer2 itself). Legitimate repetition
  measured 53–65%, so the default is 0.7.
- **`decontam`:** Every match was a verbatim benchmark question: MMLU
  questions in Open-Platypus (7 rows), MetaMathQA (5), Dolly (1), and a
  Capybara DPO set (1). No false positives were found.
- **`dedupe --near`:** It dropped 5–6% of the glaive tool-calling sets,
  which are highly templated, and 1.5% of Multilingual-Thinking, which
  repeats each prompt with answers in several languages. Elsewhere it
  dropped at most 0.4%.
- **`script` (`hangul=0.3`, opt-in):** It matched 0–4% of Korean datasets,
  mostly rows left largely in English.

The built-in phrase lists and the ways `repetition` and `script` measure
text may be tuned in minor releases to cut false positives. The changelog
lists such changes under "Changed output" ([stability.md](stability.md#quality-rules)).
