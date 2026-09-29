# convmerge

[![PyPI](https://img.shields.io/pypi/v/convmerge.svg)](https://pypi.org/project/convmerge/)
[![Python versions](https://img.shields.io/pypi/pyversions/convmerge.svg)](https://pypi.org/project/convmerge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/snowmuffin/convmerge/actions/workflows/ci.yml/badge.svg)](https://github.com/snowmuffin/convmerge/actions/workflows/ci.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/convmerge.svg)](https://pypi.org/project/convmerge/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

> **Convert Alpaca, ShareGPT, tool-calling, preference, and mixed chat datasets into one training-ready JSONL.**  
> Fetch from HuggingFace or GitHub, normalize messy Parquet / JSON / JSONL, convert [60+ popular dataset layouts](#tested-datasets) into `messages` (SFT) or `{prompt, chosen, rejected}` (DPO) rows, weighted-mix multiple domain sources, deduplicate, filter, and decontaminate — one command each, or the whole pipeline from a reproducible recipe.

`convmerge` is a **data-preparation CLI and library** for LLM supervised fine-tuning (SFT).
It takes heterogeneous instruction-tuning datasets — **Alpaca**, **ShareGPT**, raw chat JSONL,
template-rendered `text` columns, **tool-calling** data (OpenAI, Hermes, Glaive, xLAM,
LLaMA-Factory), **preference** data (UltraFeedback, HH-RLHF, Orca DPO pairs, Arena),
Parquet dumps — and produces a single clean JSONL file in the standard `messages` format
(or `alpaca` shape, or DPO preference pairs) that TRL, LLaMA-Factory, axolotl, and other
fine-tuning frameworks consume directly.

It is intentionally scoped to the **pre-training-loop** step: no model
loading, no inference, no labeling, no training orchestration. See
[Out of scope](#out-of-scope) below.

**Repository:** [github.com/snowmuffin/convmerge](https://github.com/snowmuffin/convmerge)  
**Status:** stable since 1.0: the CLI, the Python API, and the file formats
change incompatibly only in a new major version ([stability.md](docs/stability.md)).

## Quickstart

Three rows in three layouts (Alpaca, ShareGPT, OpenAI messages) become one
training file:

```bash
pip install convmerge
cat > raw.jsonl <<'JSONL'
{"instruction": "Translate to French", "input": "Good morning", "output": "Bonjour"}
{"conversations": [{"from": "human", "value": "What is 2+2?"}, {"from": "gpt", "value": "4"}]}
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
JSONL
convmerge convert -i raw.jsonl -o train.jsonl --from auto
# read 3 lines, wrote 3 examples
head -n 1 train.jsonl
# {"messages": [{"role": "user", "content": "Translate to French\nGood morning"}, {"role": "assistant", "content": "Bonjour"}]}
```

A dataset from the Hub (`fetch` needs the `[fetch-all]` extra):

```bash
pip install "convmerge[fetch-all]"
convmerge fetch hf://tatsu-lab/alpaca --max-rows 1000 -o raw
convmerge convert -i raw/tatsu-lab_alpaca.jsonl -o train.jsonl --from auto
```

A file saved in another encoding (cp949 on Korean Windows, for example) is
read with `--encoding cp949`; the output is always UTF-8.

## Install

```bash
pip install convmerge                    # core: convert, dedupe, filter, decontam; normalize for .json/.jsonl
pip install "convmerge[all]"             # full CLI: fetch (HF+GitHub), parquet, YAML presets, tokens, near dedupe
```

Granular extras:

```bash
pip install "convmerge[fetch]"           # YAML manifests + GitHub (PyYAML)
pip install "convmerge[fetch-all]"       # fetch + HuggingFace (``datasets``)
pip install "convmerge[fetch-hf]"        # same dependencies as ``fetch-all`` (backward-compatible name)
pip install "convmerge[parquet]"         # Parquet input for ``normalize``
pip install "convmerge[preset]"          # YAML convert presets (`--preset`, `preset validate`)
pip install "convmerge[tokens]"          # `tokens`: lengths + chat-template checks (transformers, no PyTorch)
pip install "convmerge[quality]"         # `dedupe --near`: near-duplicate removal (datasketch)
```

| Command / feature | Extra |
|-------------------|--------|
| `convert`, `dedupe`, `filter`, `decontam` (local eval files), `turns`, `split`, `llamafactory-info`, `axolotl-config` | *(core)* |
| `normalize` on `.parquet` | `[parquet]` |
| `fetch` with YAML manifest or GitHub | `[fetch]` |
| `fetch` with HuggingFace manifest entries | `[fetch-all]` or `[fetch-hf]` |
| `convert --preset`, `preset` | `[preset]` |
| `tokens` | `[tokens]` |
| `dedupe --near` | `[quality]` |
| `decontam --against hf:...` | `[fetch-all]` or `[fetch-hf]` |
| Everything above | `[all]` |

Or from a clone:

```bash
git clone https://github.com/snowmuffin/convmerge.git
cd convmerge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,all]"
```

## From datasets to a training run

| Trainer | Guide | Formats |
|---------|-------|---------|
| TRL (`SFTTrainer`, `DPOTrainer`) | [docs/guides/trl.md](docs/guides/trl.md) | `messages`, `preference` |
| LLaMA-Factory | [docs/guides/llamafactory.md](docs/guides/llamafactory.md) | `sharegpt`, `sharegpt-preference` + `llamafactory-info` |
| axolotl | [docs/guides/axolotl.md](docs/guides/axolotl.md) | `messages`, `sharegpt`, `preference` + `axolotl-config` |

Each guide is one recipe — fetch, convert, mix, dedupe, a `tokens` filter for
the target model's chat template and length, and a train/validation `split` —
plus the trainer config. All three were run end to end (SFT and DPO).
Korean data: [docs/guides/korean.md](docs/guides/korean.md) (한국어 가이드).

## Tested datasets

Every dataset below is converted in the test suite from a record with its exact
layout, and `python scripts/datasets.py check` runs the same conversions on real
rows from the Hub. The [Datasets workflow](.github/workflows/datasets.yml) runs
that check weekly and on pull requests that touch the catalog or the adapters.
For example:

```bash
convmerge fetch hf://NousResearch/hermes-function-calling-v1 --config func_calling --max-rows 1000 -o raw
convmerge convert -i raw/NousResearch_hermes-function-calling-v1.jsonl -o tools.jsonl --from auto --format messages

convmerge fetch hf://HuggingFaceH4/ultrafeedback_binarized --split train_prefs -o raw
convmerge convert -i raw/HuggingFaceH4_ultrafeedback_binarized.jsonl -o dpo.jsonl --from auto --format preference
```

<!-- datasets:start -->
| Dataset | Kind | Lang | Layout | `convmerge convert` flags |
|---------|------|------|--------|---------------------------|
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | SFT | en | messages | `--from auto --format messages` |
| [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | SFT | en | messages | `--from auto --format messages` |
| [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | SFT | en | messages | `--from auto --format messages` |
| [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) (gated) | SFT | en | messages | `--from auto --format messages` |
| [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | SFT | en | sharegpt | `--from auto --format messages` |
| [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | SFT | en | sharegpt | `--from auto --format messages` |
| [Magpie-Align/Magpie-Pro-300K-Filtered](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-300K-Filtered) | SFT | en | sharegpt | `--from auto --format messages` |
| [LDJnr/Capybara](https://huggingface.co/datasets/LDJnr/Capybara) | SFT | en | input/output turns | `--from auto --format messages` |
| [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | SFT | en | alpaca | `--from auto --format messages` |
| [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | SFT | en | alpaca | `--from auto --format messages` |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | SFT | en | instruction/context/response | `--from auto --format messages` |
| [garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) | SFT | en | alpaca | `--from auto --format messages` |
| [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | SFT | en | query/response | `--from auto --format messages` |
| [microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | SFT | en | question/answer | `--from auto --format messages` |
| [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | SFT | en | prompt/response | `--from auto --format messages` |
| [beomi/KoAlpaca-v1.1a](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a) | SFT | ko | alpaca | `--from auto --format messages` |
| [kyujinpy/KOR-OpenOrca-Platypus-v3](https://huggingface.co/datasets/kyujinpy/KOR-OpenOrca-Platypus-v3) | SFT | ko | alpaca | `--from auto --format messages` |
| [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) | SFT | en | text (### Human:) | `--from auto --format messages` |
| [OpenAssistant/oasst_top1_2023-08-25](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) | SFT | en | text (ChatML) | `--from auto --format messages` |
| [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k) | SFT | en | text (Llama 2) | `--from auto --format messages` |
| [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | SFT | en | sharegpt + image | `--from auto --format messages` |
| [nlpai-lab/kullm-v2](https://huggingface.co/datasets/nlpai-lab/kullm-v2) | SFT | ko | Alpaca + id | `--from auto --format messages` |
| [maywell/koVast](https://huggingface.co/datasets/maywell/koVast) | SFT | ko | ShareGPT (multi-turn) | `--from auto --format messages` |
| [heegyu/open-korean-instructions](https://huggingface.co/datasets/heegyu/open-korean-instructions) | SFT | ko | `text` with `<usr>` / `<bot>` / `<sys>` | `--from auto --format messages` |
| [FreedomIntelligence/sharegpt-korean](https://huggingface.co/datasets/FreedomIntelligence/sharegpt-korean) | SFT | ko | ShareGPT (JSON array) | `--from auto --format messages` |
| [microsoft/orca-agentinstruct-1M-v1](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1) | SFT | en | messages as a JSON string | `--from auto --format messages` |
| [nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | SFT | en | problem / generated_solution | `--from auto --format messages` |
| [facebook/natural_reasoning](https://huggingface.co/datasets/facebook/natural_reasoning) | SFT | en | question + responses[] list | `--from map --format messages --adapter-kwargs '{"map":{"user":"question","assistant":"responses[0].response"}}'` |
| [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | SFT | multi | inputs / targets | `--from auto --format messages` |
| [CertifiedJoon/Korean-Instruction](https://huggingface.co/datasets/CertifiedJoon/Korean-Instruction) | SFT | ko | Instruction / Response (capitalised) | `--from auto --format messages` |
| [heegyu/open-korean-instructions-v20231020](https://huggingface.co/datasets/heegyu/open-korean-instructions-v20231020) | SFT | ko | ShareGPT, `input` turn = system prompt | `--from auto --format messages` |
| [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | Reasoning | en | messages, inline `<think>` | `--from auto --format messages` |
| [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | Reasoning | en | ShareGPT, inline `<think>` | `--from auto --format messages` |
| [simplescaling/s1K-1.1](https://huggingface.co/datasets/simplescaling/s1K-1.1) | Reasoning | en | question / trace / attempt columns | `--from auto --format messages --adapter-kwargs '{"chat":{"output_keys":["deepseek_attempt"],"record_reasoning_keys":["deepseek_thinking_trajectory"]}}'` |
| [nvidia/Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) | Reasoning | en | `input` turns + `output` | `--from auto --format messages` |
| [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) | Reasoning | multi | messages + `thinking` field (gpt-oss) | `--from auto --format messages --reasoning thinking` |
| [a-m-team/AM-Thinking-v1-Distilled](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled) | Reasoning | en | ShareGPT, `<think>` + `<answer>` | `--from auto --format messages` |
| [nvidia/Nemotron-SFT-Instruction-Following-Chat-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v3) | Reasoning | en | messages + `reasoning_content`, first prompt withheld (`null`) | `--from auto --format messages --leading-assistant drop` |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Tool calling | en | Glaive system + chat | `--from auto --format messages` |
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Tool calling | en | Hermes <tool_call> tags | `--from auto --format messages` |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (gated) | Tool calling | en | xLAM query/answers/tools | `--from auto --format messages` |
| [llamafactory/glaive_toolcall_en](https://huggingface.co/datasets/llamafactory/glaive_toolcall_en) | Tool calling | en | LLaMA-Factory function_call | `--from auto --format messages` |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Tool calling | en | [Func(k=v)] calls, functions in the system prompt | `--from auto --format messages` |
| [Locutusque/function-calling-chatml](https://huggingface.co/datasets/Locutusque/function-calling-chatml) | Tool calling | en | function-call / function-response turns | `--from auto --format messages` |
| [younissk/tool-calling-mix](https://huggingface.co/datasets/younissk/tool-calling-mix) | Tool calling | en | `messages_json` / `tools_json` / `target_json` strings (no-call and ToolBench ReAct rows drop) | `--from auto --format messages` |
| [ZeroAgency/gemma3-pythonic-function-tool-calling-v1](https://huggingface.co/datasets/ZeroAgency/gemma3-pythonic-function-tool-calling-v1) | Tool calling | en, ru | Gemma-rendered `conversation`, Python-style calls | `--from auto --format messages` |
| [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | Preference | en | prompt + chosen/rejected lists | `--from auto --format preference` |
| [trl-lib/ultrafeedback_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) | Preference | en | chosen/rejected lists | `--from auto --format preference` |
| [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) | Preference | en | Human:/Assistant: transcripts | `--from auto --format preference` |
| [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) | Preference | en | system/question + chosen/rejected strings | `--from auto --format preference` |
| [argilla/distilabel-capybara-dpo-7k-binarized](https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized) | Preference | en | chosen/rejected lists | `--from auto --format preference` |
| [llamafactory/DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k) | Preference | en, zh | LLaMA-Factory ranking | `--from auto --format preference` |
| [lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) (gated) | Preference | en | Arena conversation_a/b + winner | `--from auto --format preference` |
| [maywell/ko_Ultrafeedback_binarized](https://huggingface.co/datasets/maywell/ko_Ultrafeedback_binarized) | Preference | ko | prompt / chosen / rejected strings | `--from auto --format preference` |
| [kuotient/orca-math-korean-dpo-pairs](https://huggingface.co/datasets/kuotient/orca-math-korean-dpo-pairs) | Preference | ko | system / question / chosen / rejected | `--from auto --format preference` |
| [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | Preference | en | response_0/1 + better_response_id | `--from map --format preference --adapter-kwargs '{"map":{"user":"prompt","responses":["response_0","response_1"],"preferred":"better_response_id"}}'` |
| [stanfordnlp/SHP](https://huggingface.co/datasets/stanfordnlp/SHP) | Preference | en | human_ref_A/B + labels | `--from map --format preference --adapter-kwargs '{"map":{"user":"history","responses":["human_ref_A","human_ref_B"],"preferred":"labels","preferred_values":{"1":0,"0":1}}}'` |
| [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | Preference | multi | context turns + response1/2 + overall_preference | `--from map --format preference --adapter-kwargs '{"map":{"turns":"context","responses":["response1","response2"],"preferred":"overall_preference","preferred_values":{"-3":0,"-2":0,"-1":0,"1":1,"2":1,"3":1}}}'` |
| [argilla/distilabel-math-preference-dpo](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo) | Preference | en | instruction + chosen_response / rejected_response | `--from auto --format preference` |
| [shibing624/DPO-En-Zh-20k-Preference](https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference) | Preference | en, zh | history pairs + question + response_chosen / response_rejected | `--from auto --format preference` |
| [ChuGyouk/argilla-distilabel-math-preference-dpo-korean](https://huggingface.co/datasets/ChuGyouk/argilla-distilabel-math-preference-dpo-korean) | Preference | ko | English + `_ko` columns, chosen_response / rejected_response | `--from map --format preference --adapter-kwargs '{"map":{"user":"instruction_ko","chosen":"chosen_response_ko","rejected":"rejected_response_ko"}}'` |
<!-- datasets:end -->

A preference dataset can also feed SFT: `--format messages --preference chosen`
trains on the chosen answers.

## Commands

### 1. `fetch` — pull raw data from HF + GitHub via a YAML manifest

> HuggingFace entries delegate to `datasets.load_dataset(...).to_json(...)`,
> i.e. the output is a **JSONL dump** of the selected split. GitHub entries
> support a single raw URL, recursive Trees API fetch with an extension
> filter, or `git clone` (with optional `git lfs pull`). `fetch` is a
> reproducible downloader, not a mirror of HuggingFace's Arrow cache.

```yaml
# manifest.yaml
version: 1
defaults: { output_root: ./raw, resume: true }
auth:     { hf_token_env: HF_TOKEN, github_token_env: GITHUB_TOKEN }
datasets:
  - { name: alpaca-ko, hf: MarkrAI/KoCommercial-Dataset, split: train }
  - { name: orca-raw,
      url: https://raw.githubusercontent.com/org/repo/main/data/train.jsonl }
  - { name: repo-tree,
      url: https://github.com/org/example-repo, ext: [".jsonl"] }
  - { name: big-lfs,
      url: https://github.com/org/big-lfs-repo, mode: clone, lfs: true }
```

```bash
convmerge fetch manifest.yaml -o ./raw
# or one-shot shortcuts:
convmerge fetch hf://org/dataset -o ./raw --split train
convmerge fetch https://github.com/org/repo -o ./raw --ext .jsonl
```

Tokens resolve in order CLI flag → file → env var, and are redacted from logs.
See [docs/fetch.md](docs/fetch.md) for the full schema.

### 2. `normalize` — reshape parquet / messy JSON into clean JSONL

```bash
convmerge normalize -i ./raw -o ./jsonl
```

Handles parquet (streamed via `pyarrow`), top-level JSON arrays, concatenated
single-line JSON (`{...}{...}{...}`), JSONL whose lines are arrays (wrapped as
`{"conversation": [...]}`), and already-valid JSONL. A directory input is
walked recursively and mirrored under the output directory.

### 3. `convert` — adapter + emitter pipeline

```bash
convmerge convert -i ./jsonl/alpaca.jsonl -o ./train/alpaca.messages.jsonl \
  --from alpaca --format messages

convmerge convert -i ./jsonl/mixed.jsonl -o ./train/mixed.messages.jsonl \
  --from auto --format messages         # auto-detecting chat adapter

# Optional: YAML preset (pip install "convmerge[preset]")
convmerge preset init -o convert_preset.yaml
convmerge preset validate convert_preset.yaml
convmerge convert -i ./jsonl/mixed.jsonl -o ./out.jsonl --preset convert_preset.yaml
```

Adapters: `alpaca`, `sharegpt`, `chat` (alias `auto`), and `map` for any other
layout (dotted paths such as `dialogue[].utterances[]`, no code; see
[docs/format.md](docs/format.md#field-mapping---from-map)).  
Output formats: `messages`, `alpaca`, `preference` (DPO pairs), `sharegpt` and
`sharegpt-preference` (LLaMA-Factory / Unsloth).

> **Tool calling and multimodal:** OpenAI `tool_calls` / `tools`, LLaMA-Factory
> `function_call` / `observation` turns, Hermes `<tool_call>` tags, Glaive and
> xLAM layouts all come out as standard `tool_calls` / `tools`, and image /
> audio / video references (`image_url` parts, `images` columns with `<image>`
> tokens) are preserved in the `messages` output. Media is kept by reference only — convmerge never
> downloads or decodes it. `--from sharegpt` keeps whole conversations since
> 0.6.0 (`turn_mode: pairs` restores the old split). See
> [docs/format.md](docs/format.md#sharegpt).

Preference (DPO / reward) datasets: `--format preference` writes
`{prompt, chosen, rejected}` pairs for TRL's `DPOTrainer`; `--preference chosen`
trains SFT on the chosen answer instead (UltraFeedback, HH-RLHF, Orca DPO pairs,
LLaMA-Factory ranking, TRL, and Chatbot Arena shapes).

Rendered `text` columns (ChatML, Llama 2/3, Gemma, `### Human:`, HH, the Alpaca
prompt) are split back into turns by `--from auto`.

Reasoning data: traces stored inline (`<think>...</think>`) or in a
`reasoning_content` / `thinking` field are kept; `--reasoning` moves them to
the field your model's chat template reads (`reasoning_content` for Qwen3 /
DeepSeek, `thinking` for gpt-oss), inline, or drops them. See
[docs/guides/reasoning.md](docs/guides/reasoning.md).

Strict chat templates: `--system fold` (no system role), `--merge-consecutive`
("roles must alternate"), `--reasoning-turns last` / `--split-turns`
(templates that render reasoning only after the last user turn). Tool-call
turns get `"content": ""`, which every common template accepts. Symptom →
fix table: [docs/guides/troubleshooting.md](docs/guides/troubleshooting.md).

Large files: `--workers N` converts with N processes (same output and
stats as a single process; ~3.8x faster with 4 workers in our benchmark).
`filter` and `dedupe --near` take `--workers N` too.

Every example is validated before it is written; ones with no user turn,
empty messages, or unmatched tool results are dropped and counted by reason
(`--on-invalid keep|fail` to change that, `--report PATH` for details,
`convmerge validate -i FILE` to check an existing file). See
[docs/format.md](docs/format.md#validation).

Presets and team-specific tuning: [docs/custom_presets.md](docs/custom_presets.md).

> `chat` / `auto` is a **heuristic** adapter: it inspects the keys of each
> input record (`messages`, `conversation(s)`, `text`, `conversation_a`/`_b`,
> `instruction`/`input`/`output`, …) and routes to the right branch with a
> configurable role map. For unusual schemas, pin an explicit adapter
> (`alpaca`, `sharegpt`) or override keys programmatically — see
> [docs/format.md](docs/format.md).

### 4. `mix` — domain-controlled weighted merge

```bash
# Inline weights
convmerge mix \
  -i ./train/code.messages.jsonl:0.4 \
     ./train/math.messages.jsonl:0.3 \
     ./train/general.messages.jsonl:0.3 \
  -o ./train/mixed.jsonl --total 100000 --seed 42

# Or via a config file (YAML requires convmerge[preset])
convmerge mix mix.yaml
```

```yaml
# mix.yaml
seed: 42
total: 100000
output: ./train/mixed.jsonl
sources:
  - { path: ./train/code.messages.jsonl,    weight: 0.4 }
  - { path: ./train/math.messages.jsonl,    weight: 0.3 }
  - { path: ./train/general.messages.jsonl, weight: 0.3 }
```

Weights are normalized automatically and need not sum to 1.0. When a source
has fewer records than its allocation it is clipped; pass `--oversample` to
repeat records instead. `mix` streams: it reads each source twice and shuffles
through temporary files next to the output, so memory stays small even when
merging multi-GB sources (`--sampler v1` reproduces mixes made before 0.7). A sidecar `.mix.json` is written alongside
the output recording the exact seed, weights, and per-source counts for full
reproducibility. Omit `--total` to merge all records from every source.

### 5. `dedupe` / `filter` / `decontam` / `tokens` / `split` — ready for training

```bash
convmerge dedupe -i ./train/mixed.jsonl -o ./train/mixed.dedup.jsonl
# --near also drops near-copies (MinHash LSH; needs convmerge[quality])

# Rule-based quality filter: refusals and "as an AI language model" disclaimers
# (English, Korean), empty answers, looping answers or reasoning traces, and
# preference pairs whose answers are the same or whose rejected side is empty.
# Optional: --enable slop, --min-chars/--max-chars, --min-script hangul=0.3.
# Prints a JSON report with example rows per rule; -o keeps the rows that pass.
convmerge filter -i ./train/mixed.dedup.jsonl -o ./train/clean.jsonl --rejects rejected.jsonl

# Drop rows that share a 13-word n-gram with a benchmark (prompts; --check all
# for answers too). hf: sources need convmerge[fetch-all].
convmerge decontam -i ./train/clean.jsonl --against hf:openai/gsm8k:main \
  --against hf:cais/mmlu:all -o ./train/decontam.jsonl

# Render every row with the model's chat template: length percentiles, rows the
# template rejects (with line numbers), double-encoded tool arguments, answers
# that start beyond --max-tokens, answers not followed by a stop token, reasoning
# the template drops, {% generation %} support, and the instruction/response
# markers for Unsloth's train_on_responses_only -- with the convert flag that
# fixes each ("hints"). -o keeps the rows that render and fit.
# Needs convmerge[tokens] (transformers, no PyTorch).
convmerge tokens -i ./train/decontam.jsonl --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --max-tokens 4096 -o ./train/fit.jsonl

# Train/validation split by content hash: reproducible, order-independent,
# duplicates never straddle the two sides. --val-rows N for an exact count.
convmerge split -i ./train/fit.jsonl -o ./train/train.jsonl --val 0.02   # + train.val.jsonl

# LLaMA-Factory: add the dataset_info.json entry for a --format sharegpt file.
convmerge llamafactory-info -i ./data/sft.jsonl --name my_sft --info ./data/dataset_info.json

# axolotl: the datasets: block whose field names match a converted file.
convmerge axolotl-config -i ./train/train.jsonl --val ./train/train.val.jsonl

# Single-turn vs multi-turn report (and split)
convmerge turns -i ./train/train.jsonl --single-out single.jsonl --multi-out multi.jsonl
```

See [docs/quality.md](docs/quality.md) for the `filter` rules, `decontam`, and
near dedupe, [docs/format.md](docs/format.md) for adapter / emitter schemas,
[docs/fetch.md](docs/fetch.md) for manifest details, and
[docs/api.md](docs/api.md) for the Python API and writing plugins
(custom adapters / output formats via entry points).
[docs/stability.md](docs/stability.md) lists what stays compatible across
releases (API, CLI flags and exit codes, file formats).

### 6. `run` — the whole pipeline from one recipe

```yaml
# recipe.yaml
version: 1
output: train/mixed.jsonl
sources:
  alpaca: { path: data/alpaca_data.json, convert: { from: alpaca } }
  tools:
    fetch: { url: https://raw.githubusercontent.com/org/repo/main/tools.jsonl }
    convert: { from: sharegpt }
mix: { total: 100000, seed: 42, weights: { alpaca: 0.7, tools: 0.3 } }
dedupe: true
filter: true                                                         # optional
decontam: { against: [hf:openai/gsm8k:main] }                        # optional
tokens: { tokenizer: Qwen/Qwen2.5-7B-Instruct, max_tokens: 4096 }   # optional
split: { val: 0.02 }                                                 # optional
```

```bash
convmerge run recipe.yaml --plan     # what would run, and why
convmerge run recipe.yaml            # fetch → normalize → convert → mix → dedupe → filter → decontam → tokens → split
convmerge run recipe.yaml --frozen   # CI: fail unless the lock file is current
```

Each step is the same command you would type by hand, so the result is
identical. `recipe.lock.json` records options, convmerge version, and
input/output digests; the next run repeats only the steps whose inputs or
options changed. See [docs/recipes.md](docs/recipes.md).

## Out of scope

To keep the package lean and dependency-free at its core, `convmerge` does
**not** include — and has no plans to include — the following:

- **Model loading / inference / training.** No PyTorch, Transformers, vLLM,
  or similar runtime is imported by the core or any shipped extra.
- **Model-based labeling or classification of samples** (e.g. topic tagging,
  LLM-as-judge or classifier quality scores, safety classification). These
  are left to upstream tools or private pipelines; `filter` covers the
  deterministic, rule-based checks.
- **RLHF / DPO / preference-dataset construction** beyond passing through
  existing pairwise rows via the `chat` adapter's `pairwise_mode`.
- **Training-job orchestration** (SkyPilot, RunPod, Modal, K8s operators).
- **Prompt templating / chat-template rendering** for specific model
  families. Output JSONL uses the standard `messages` / `alpaca` shapes;
  downstream trainers apply their own template.
- **Tokenizer-aware length filtering, packing, or curriculum scheduling.**
  Those live in the training stack, not here.
- **Downloading, decoding, or transforming media.** Images, audio, and video
  are carried through as references (URLs or paths) exactly as the source
  gave them; fetching and preprocessing the files is the trainer's job.
- **Scraping HTML pages or running browser automation.** Structured JSON /
  JSONL / Parquet inputs only.

If any of these are important to your workflow, wire `convmerge` in as one
step of a larger pipeline rather than expecting it to grow into those areas.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide — setup, local
checks, code conventions, and a walkthrough for adding a new adapter /
emitter. CI runs Ruff, mypy, and pytest on Python 3.10 – 3.12.

```bash
pip install -e ".[dev,all]"
ruff check src tests
ruff format --check src tests
mypy
pytest -q
```

Participation in this project is governed by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

Good first PRs: new adapters / emitters for public dataset schemas, new
fetch backends (GitLab / Zenodo / Kaggle), recipe examples under
[`examples/`](examples/), and docs improvements. Browse the
[`good first issue`](https://github.com/snowmuffin/convmerge/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
label for concrete starting points.

## PyPI release (maintainers)

Releases run from [`.github/workflows/publish.yml`](.github/workflows/publish.yml)
on pushing a `v*` tag. Publishing authenticates via the **`PYPI_API_TOKEN`**
GitHub Actions secret.

1. Create an API token on [pypi.org](https://pypi.org/manage/account/token/).
   - If the project already exists on PyPI, scope the token to the
     `convmerge` project (principle of least privilege).
   - For the very first upload (project not yet registered), PyPI does not
     allow project-scoped tokens — use **Entire account** scope for the
     first release, then rotate to a project-scoped token afterwards and
     revoke the original.
2. In the GitHub repo, *Settings → Secrets and variables → Actions → New
   repository secret*, add `PYPI_API_TOKEN` with the token value.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.

## Changelog

[CHANGELOG.md](CHANGELOG.md) · upgrading: [to 1.0](docs/migration-1.0.md), [from 0.6](docs/migration-0.7.md), [from 0.5](docs/migration-0.6.md)

## License

MIT
