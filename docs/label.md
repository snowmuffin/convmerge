# Labeling with a model: `convmerge label`

`label` asks a model one question about each row and writes the answer to
the row's `meta`. The answer is either one class from a list (`--classes`)
or one point on an integer scale (`--scale`). No row is dropped. Other steps
then use the labels:

- `report` shows their distribution;
- `order --by meta:NAME` orders rows by a scale label;
- `filter --meta-min NAME=N` applies a threshold you have a reason for.

convmerge ships no built-in "quality judge" prompt. In the
[score-filter study](score-filter.md), keeping only the best human-rated rows
did no better than random rows, so a label is a way to see and steer what
the data holds, not a quality filter. The design is in
[design/labeling.md](design/labeling.md).

```bash
# Topic, from a model served locally (vLLM, Ollama, LM Studio, TGI, ...)
convmerge label -i train.jsonl -o labeled.jsonl \
  --endpoint http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --name topic --classes code,math,writing,other \
  --question "What is this conversation mainly about?"

# Difficulty on a 1-5 scale, with a rubric in a file
convmerge label -i labeled.jsonl -o labeled2.jsonl \
  --endpoint http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --name difficulty --scale 1-5 --rubric difficulty.md
```

## Where rows are sent

The endpoint is any server that speaks the OpenAI chat-completions protocol
(`POST {endpoint}/chat/completions`). Every request is sent with:

- `temperature` 0;
- a small `max_tokens`;
- `seed` 0.

Requests ask for no JSON mode, guided decoding or tool calls.

- **Hosts.** Only local and private hosts are allowed by default:
  `localhost`, `127.0.0.0/8`, `::1`, `10/8`, `172.16/12`, `192.168/16`,
  `fc00::/7`, link-local addresses and `*.local`. Any other host needs
  `--allow-external`. Without it, the run stops before anything is sent.
- **Rows go as they are.** Remove personal data before labeling with an
  external API.
- **API keys.** `--api-key-env NAME` reads the key from that environment
  variable. The key is never written to the report, the cache or an error
  message.

## What the model sees

Each request has one system turn and one user turn. The user turn renders
the conversation as plain text:

- one `role: content` block per turn;
- tool calls as `[tool call] name(arguments)`;
- images, audio and video as `[image]`, `[audio]` and `[video]`.

A preference row is shown as its prompt and its chosen answer.

A conversation longer than `--max-chars` (default 8,000) is cut from the
middle. The start and the end are kept, and the row is counted as
`truncated`.

The built-in prompts:

- **classes:** the question, the class list, the conversation, then "Answer
  with exactly one class name from the list and nothing else."
- **scale:** the scale, the rubric, the conversation, then "Answer with one
  integer from LO-HI and nothing else."

`--prompt FILE` replaces the user turn with your own
[`string.Template`](https://docs.python.org/3/library/string.html#template-strings).
It must contain `$conversation`, and may also use `$classes`, `$scale`,
`$question` and `$rubric`. Write `$$` for a literal `$`.

## How answers are read

Parsing is strict and the same for every server:

- **classes:** the first non-empty line, with whitespace, quotes, `*` and a
  final period trimmed, must equal a class name (case-insensitive). The
  label is the class name as you wrote it.
- **scale:** the first non-empty line must be one integer in the range. The
  label is that integer.

An answer that does not parse is asked once more. The retry quotes the
answer and restates the format. If the second answer fails too, the label is
`null` and the row counts as a parse failure. A row that gets no answer at
all is also labeled `null`: this happens after the retries for connection
errors, timeouts, HTTP 429 and 5xx, or under `--cache-only`.

## Output

Each row is written with `meta[NAME]` set. `meta` is created if missing, and
`--meta-key KEY` writes under another key. Every other field is kept.

- **Existing labels.** If any row already has `meta[NAME]`, the run refuses
  before making a call. Pass `--overwrite` to replace those labels.
- **Rows without a label.** These rows are written as they were and counted
  as `unreadable`: invalid JSON, no conversation, or a `meta` that is not an
  object.

The JSON report on stdout has the following fields:

- the endpoint host (never the full URL or key) and the model;
- the `model` and `system_fingerprint` the server reported;
- the prompt template's SHA-256 and the generation parameters;
- counts: `rows`, `labeled`, `truncated`, `unreadable`, `parse_failures`,
  `call_failures`, `calls`, `cache_hits`, `retries`;
- `values`: a count per class, or per scale point;
- `failed_samples`: up to three answers that did not parse.

## Cache, budget and stops

Answers are kept in a SQLite file, by default
`.convmerge-cache/llm.sqlite` next to the output; `--cache FILE` picks
another. The key is the host, the model, the rendered prompt and the
generation parameters. It does not include the row's own bytes, so the same
conversation in another file reuses the answer.

- **Re-runs.** A cached request is never sent again. Re-running the same
  command makes no calls and writes the same file.
- **Offline runs.** `--cache-only` makes no calls at all. Rows not in the
  cache get `null`.
- **Concurrency.** `--workers` (default 4) requests run at once, each with a
  `--timeout` (default 60 s). Connection errors, timeouts, HTTP 429 and 5xx
  are retried with backoff, up to 3 attempts.

The run stops with exit code 1, and writes no output file, in these cases:

- the server answers 401, 403 or 404 (a wrong key, model or URL);
- the first 20 calls all fail;
- `--max-calls N` network calls have been made.

Every answer that came back before the stop stays in the cache, so the next
run continues from there.

Exit code 2 means a bad invocation:

- an option error;
- an external host without `--allow-external`;
- existing labels without `--overwrite`.

## Python

```python
from convmerge import label_jsonl

stats = label_jsonl(
    "train.jsonl", "labeled.jsonl",
    endpoint="http://localhost:8000/v1", model="Qwen/Qwen2.5-7B-Instruct",
    name="topic", classes=["code", "math", "writing", "other"],
    question="What is this conversation mainly about?",
    cache=".convmerge-cache/llm.sqlite",
)
print(stats.to_report()["values"])
```

`rubric` and `prompt` take text, not file paths. Without `cache`, nothing is
cached. A stop raises `convmerge.llm.LLMError`; `BudgetExceeded` is a
subclass of it.
