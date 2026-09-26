# Golden fixtures

Realistic, hand-written samples of the dataset shapes convmerge meets in the
wild. The content is made up, but the structure matches each source format.

| Fixture | Shapes |
|---------|--------|
| `openai_chat` | OpenAI chat fine-tuning rows: system prompt, `tool_calls` + `role: tool`, top-level `tools`, text-part content, `name` |
| `openai_multimodal` | OpenAI vision (`image_url` parts); TRL/HF style `{"type": "image"}` placeholder + `images` column |
| `sharegpt_basic` | `conversations` / `from` / `value`: single-turn, multi-turn with system, non-alternating turns, empty system |
| `sharegpt_tools` | LLaMA-Factory function calling: `function_call` / `observation` turns, `system` and `tools` (JSON string) columns |
| `sharegpt_multimodal` | LLaMA-Factory `images` column with `<image>` tokens; LLaVA `image` field |
| `alpaca` | instruction / input / output, `response` alias, LLaMA-Factory `system` + `history` |
| `chat_variants` | `conversation` key, plain `text`, pairwise `conversation_a/b` with winner or tie, `question`/`answer`, `prompt`/`response` with stray `text`, `messages` using `from`/`value` |
| `messy` | blank line, invalid JSON, non-object rows, unmappable object, empty and assistant-only conversations |

`expected/` holds the convert output for each case in `tests/test_golden.py`
(`<fixture>.<adapter>[-options].<format>.jsonl`) and its drop counters
(`.stats.json`). They record **current** behavior, including known gaps
(tool calls, multimodal content, and LLaMA-Factory columns are not yet
preserved). An intended behavior change updates these files in the same
commit; see CONTRIBUTING.md.
