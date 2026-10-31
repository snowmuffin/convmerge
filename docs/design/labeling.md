# Design: model labeling (`convmerge label`)

Status: **draft design**. Changes after merge are made by later PRs that cite
this page. Implemented: the model call layer (`src/convmerge/llm.py`) and
`convmerge label` ([docs/label.md](../label.md)). Not yet: the recipe
`label:` step, the `report` section and `mix --group-by`.

## Why, and what changed

The improvement plan gated model-based features on the
[score-filter study](../score-filter.md). That study found that training on
the best human-rated rows gave the same eval loss as random rows at the same
token count, so the gate failed. As the study's design fixed before the run,
the model-features stage is reduced to **labeling**:

- a model's answer about a row is written to the row's `meta`;
- existing steps use those values (`report`, `order`, and later `mix`);
- convmerge ships **no** step that drops rows by a model score, and no
  built-in "quality judge" prompt.

`filter --meta-min` stays. It applies a threshold someone has a reason for,
whatever wrote the value, and the docs say what the study found.

## Scope

**In:**

- **Model call layer.** Talks to one OpenAI-compatible endpoint, with a
  cache, a host policy, retries and a budget.
- **`convmerge label`** and the recipe `label:` step:
  - closed-set classification (`--classes`);
  - integer scales (`--scale`).
- **Consumers.** `report` shows label distributions; `order --by meta:FIELD`
  already works. Later, `mix` weights by label group (in 1.10).

**Out:**

- Embedding-based dedupe. It would use the same call layer, so it is decided
  again once labeling is in use.
- Preference-pair generation, score filtering, built-in judge prompts.
- Running models in-process. Local GPUs are reached through a server (vLLM,
  Ollama, LM Studio, TGI) on the same protocol.

## Model call layer (`convmerge.llm`)

### Protocol

OpenAI-compatible `POST {endpoint}/chat/completions` only. The request
carries:

- `model`;
- `messages`: one system turn and one user turn;
- `temperature: 0`;
- `max_tokens` (small, set per task);
- `seed` (0 by default; `seed=None` leaves it out for a server that rejects
  it).

Nothing else is assumed: no JSON mode, no guided decoding, no tool calls.
Every server above speaks this subset.

HTTP goes through `requests`, which becomes a core dependency. It meets the
five conditions in CONTRIBUTING, and this page is the proposal that
CONTRIBUTING asks for. It is declared with a lower bound only and sits behind
a small internal interface (`_post_json(url, body, headers, timeout)`), so it
can be replaced without touching callers.

### Hosts and credentials

- **Allowed by default:** loopback (`localhost`, `127.0.0.0/8`, `::1`),
  private ranges (`10/8`, `172.16/12`, `192.168/16`, `fc00::/7`), link-local,
  and `*.local`.
- **Any other host** needs `--allow-external` (recipe:
  `allow_external: true`). Without it the call is refused before anything is
  sent, with a message naming the host.
- **API keys:** an API key is read from an environment variable named by
  `--api-key-env NAME`. The value never reaches a report, a lock file, an
  error message or the cache.
- **Personal data:** the docs say plainly that rows are sent as they are, and
  that personal data must be removed before labeling with an external API.

### Cache

- **Storage:** a SQLite file in the work directory. The CLI default is
  `.convmerge-cache/llm.sqlite` next to the output; in a recipe it is under
  `workdir`.
- **Key:** SHA-256 of
  - the endpoint host;
  - the model name;
  - the rendered prompt (system and user text);
  - the generation parameters.

  The row's own bytes are not part of the key: the same conversation in
  another file reuses the answer.
- **Value:**
  - the raw response text, parsed again on every run (a change to the
    parse rules applies to cached answers);
  - the `model` and `system_fingerprint` fields the server returned, if
    any;
  - the time.
- **Reuse:** a cached key is never sent again. `--cache-only` turns a missed
  key into a row with no answer (`null`, counted as a call failure) instead
  of a call, so a run with a full cache needs no network and no endpoint.
- **Writes:** one transaction per batch of answers. An interrupted run keeps
  every answer that came back.

### Calls, retries, budget

- `--workers N` concurrent requests (default 4) and a per-request
  `--timeout` (default 60 s).
- Connection errors, timeouts, HTTP 429 and 5xx are retried with
  exponential backoff, up to 3 attempts. HTTP 401, 403 and 404 stop the run
  at once: they mean a wrong key, model or URL, and every row would fail the
  same way.
- **Early stop:** if the first 20 calls all fail, the run stops instead of
  labeling every row `null`.
- **Budget:** `--max-calls N` caps network calls; cache hits are free. When
  the cap is reached, the run stops with exit code 1, and the answers so far
  stay in the cache.
- **Plan:** `run --plan` reports, per `label:` step, the rows to label, the
  cache hits, and the calls it would make. With `price_per_call` in the
  recipe it also multiplies out a cost. There is no built-in price table;
  prices change too often.

### What is recorded

The step report and the recipe lock record:

- the endpoint host, never the full URL or key;
- the model name, and the `model` / `system_fingerprint` the server
  returned;
- the prompt template's SHA-256;
- the generation parameters;
- counts: rows, calls, cache hits, parse failures, call failures, truncated
  rows.

The reproducibility promise is the one in [stability.md](../stability.md):
given the same cache, every byte after the label step is the same. Without
the cache, a model may answer differently, and the lock shows which server,
model and prompt produced the cached answers.

## `convmerge label`

```bash
convmerge label -i train.jsonl -o labeled.jsonl \
  --endpoint http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --name topic --classes code,math,writing,other \
  --question "What is this conversation mainly about?"

convmerge label -i train.jsonl -o labeled.jsonl \
  --endpoint http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --name difficulty --scale 1-5 --rubric difficulty.md
```

### Tasks

**Classification** (`--classes a,b,c`, at least 2) answers `--question`. The
built-in prompt does three things:

- lists the classes;
- shows the conversation;
- asks for exactly one class name and nothing else.

**Scale** (`--scale LO-HI`, integers) rates the conversation against a
rubric given as text (`--rubric FILE`) or inline (`--question`). The prompt
asks for one integer in the range and nothing else.

### Custom prompts

`--prompt FILE` replaces the built-in user turn. The file is a
`string.Template`, chosen so the core needs no Jinja. Its placeholders:

- `$conversation`;
- `$classes` or `$scale`;
- `$question`;
- `$rubric`.

The template's hash is part of the cache key and of the lock. The docs show
topic and difficulty examples, and no quality-score example.

### What the model sees

The conversation is rendered as plain text, one `role: content` block per
turn, tool calls as their name and arguments, and media as `[image]`
markers.

A preference row is shown as its prompt and its chosen answer.

`--max-chars` (default 8,000) cuts the rendered conversation from the
middle, keeping the start and the end, and the row is counted as
`truncated`. This bounds the cost of very long rows.

### Parsing

Parsing is strict, and the same rule applies to every server:

- **Class:** the first non-empty line of the answer, trimmed of whitespace,
  quotes and a final period, must equal one class name (case-insensitive).
- **Scale:** the first line must be one integer in range.

Anything else is retried once, with the previous answer quoted and the format
restated. A second failure writes `null` and counts a parse failure. The row
is never dropped: whether to keep unlabeled rows is the next step's choice.

### Output

Each row is written with `meta[NAME]` set: `meta` is created if missing, and
`--meta-key` names another key. All other fields are kept as they are.

If `meta[NAME]` already exists, the command refuses to run unless
`--overwrite` is given. A label from another tool is not replaced by
accident.

Rows that no adapter can read are written as they are, without a label, and
counted.

### Report

The JSON report gives:

- rows, labeled, and truncated rows;
- calls, cache hits, parse failures and call failures;
- the distribution of values: a count per class, or a histogram for a
  scale;
- up to three samples of failed answers.

`convmerge report` shows the same for a recipe's `label:` steps.

## Recipe `label:` step

```yaml
label:
  - name: topic
    endpoint: http://localhost:8000/v1
    model: Qwen/Qwen2.5-7B-Instruct
    classes: [code, math, writing, other]
    question: What is this conversation mainly about?
  - name: difficulty
    endpoint: http://localhost:8000/v1
    model: Qwen/Qwen2.5-7B-Instruct
    scale: 1-5
    rubric: prompts/difficulty.md      # a step input
    max_calls: 20000
```

- **Position:** the stages run in this order:

  `mix` → `dedupe` → `filter` → `decontam` → `tokens` → **`label`** →
  `order` → `render` → `split`

  Labeling comes after every step that drops rows, so no call is spent on a
  row that is dropped later. It comes before `order`, so `order: {by:
  meta:difficulty}` can read the labels.
- **One step per entry.** Each list entry is its own step (`label.topic`,
  `label.difficulty`), with its own lock entry.
- **Step options:** host, model, task, classes or scale, the template hash,
  and the generation parameters. Changing any of them re-runs that step.
  Calls the cache already answered cost nothing.
- **Step inputs:** `rubric` and `prompt` files.
- **Recipe keys:**
  - `allow_external` and `api_key_env` are recipe keys, never values
    written in the lock;
  - `max_calls`, `workers` and `timeout` do not change the output, so they
    are not options and changing them does not re-run a step.

## Later: `mix` by label group (1.10)

```bash
convmerge mix -i labeled.jsonl --group-by meta:topic --weights code=0.3,math=0.3,writing=0.2,other=0.2 \
  --total 50000 --seed 0 -o mixed.jsonl
```

One file is split into groups by a `meta` value, and the groups are sampled
with the v2 sampler. This is the same code path as weighting several source
files, so the reproducibility guarantees carry over. Rows without the value
form the group `null`, which can be weighted or left out. The exact flags
are settled in that release's PR.

## Testing

- **CI** uses a fake OpenAI-compatible server (stdlib `http.server`) started
  per test. Tests cover:
  - parse rules and the retry;
  - cache hits and `--cache-only`;
  - the host policy;
  - fail-fast on 401/404 and the early stop;
  - backoff on 429/5xx;
  - the budget stop;
  - truncation and preference rows;
  - `--overwrite`;
  - recipe re-runs when options change, and none when only workers change;
  - the report;
  - keys never being written.

  No network, no API key.
- **Contract.** Snapshots pin the new CLI flags and the public names. The
  public surface is `label_jsonl` and `LabelStats` in `convmerge`;
  `convmerge.llm` stays internal until a second user of it exists.
- **Real model, documentation only.** A small public dataset with gold labels
  is labeled once through a small model server. Agreement and the parse
  success rate are recorded in the docs as a reference, not as a promise,
  because they depend on the model.

## Order of work

1. This design (docs only). Done.
2. The call layer, with the fake-server tests, and `requests` in the core.
   Done: `src/convmerge/llm.py`.
3. `label`, the recipe step, and the report section. Then release 1.9.0.
   Done: `label` (`src/convmerge/label.py`).
4. `mix --group-by` and the `run --plan` cost line. Then release 1.10.0.

## Decided

Decided on 2026-10-31, from the improvement plan:

- one protocol, OpenAI-compatible chat completions;
- a SQLite cache;
- the names `convmerge label`, `--name`, `--classes`, `--scale` and recipe
  `label:`;
- embedding dedupe left out of this stage.
