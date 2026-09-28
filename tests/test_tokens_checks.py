"""convmerge tokens: loss-mask markers, answers beyond the limit, stop tokens, reasoning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("jinja2")

from convmerge.cli import main  # noqa: E402
from convmerge.tokens import check_tokens  # noqa: E402

# ChatML-like; like Qwen3, a turn's reasoning_content is rendered only after the
# last user turn.
QWEN3ISH = (
    "{% set ns = namespace(last=-1) %}{% for m in messages %}{% if m.role == 'user' %}"
    "{% set ns.last = loop.index0 %}{% endif %}{% endfor %}"
    "{% for m in messages %}<|im_start|> {{ m.role }} "
    "{% if m.role == 'assistant' and loop.index0 > ns.last and m.reasoning_content %}"
    "<think> {{ m.reasoning_content }} </think> {% endif %}{{ m.content }} <|im_end|> "
    "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)
WITH_GENERATION = (
    "{% for m in messages %}<|im_start|> {{ m.role }} {% if m.role == 'assistant' %}"
    "{% generation %}{{ m.content }} <|im_end|>{% endgeneration %}{% else %}"
    "{{ m.content }} <|im_end|>{% endif %} {% endfor %}"
)
NO_END = "{% for m in messages %}{{ m.role }} : {{ m.content }} {% endfor %}"
CONCAT = "{% for m in messages %}{{ m.role + ' ' + m.content }} <|im_end|> {% endfor %}"


def _tokenizer(path: Path, *, eos: str | None, generation_eos: list[int] | None = None) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token=eos)
    fast.chat_template = QWEN3ISH
    fast.save_pretrained(path)
    if generation_eos is not None:
        (path / "generation_config.json").write_text(json.dumps({"eos_token_id": generation_eos}))
    return path


@pytest.fixture(scope="module")
def tok_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _tokenizer(tmp_path_factory.mktemp("tok"), eos="<|im_end|>")


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _msgs(*turns: tuple[str, str], **extra: str) -> dict:
    msgs = [{"role": r, "content": c} for r, c in turns]
    msgs[-1].update(extra)
    return {"messages": msgs}


def test_generation_tags_detected(tmp_path: Path, tok_dir: Path) -> None:
    src = _write(tmp_path / "a.jsonl", [_msgs(("user", "hi"), ("assistant", "yo"))])
    without = check_tokens(src, tokenizer=str(tok_dir))
    assert without.generation_tags is False
    assert any("{% generation %}" in h for h in without.hints())
    with_tags = check_tokens(src, tokenizer=str(tok_dir), chat_template=WITH_GENERATION)
    assert with_tags.generation_tags is True
    assert not any("generation" in h for h in with_tags.hints())


def test_answer_beyond_limit(tmp_path: Path, tok_dir: Path) -> None:
    long_prompt = " ".join(["word"] * 30)
    src = _write(
        tmp_path / "a.jsonl",
        [
            _msgs(("user", long_prompt), ("assistant", "ok")),
            _msgs(("user", "hi"), ("assistant", "ok")),
        ],
    )
    st = check_tokens(src, tokenizer=str(tok_dir), max_tokens=20)
    assert (st.over_limit, st.answer_beyond_limit) == (1, 1)
    assert st.to_report()["answer_beyond_limit"] == 1
    assert check_tokens(src, tokenizer=str(tok_dir)).answer_beyond_limit == 0  # no limit


def test_stop_tokens_and_missing_eos(tmp_path: Path) -> None:
    src = _write(tmp_path / "a.jsonl", [_msgs(("user", "hi"), ("assistant", "yo"))])
    ok = _tokenizer(tmp_path / "ok", eos="<|im_end|>")
    st = check_tokens(src, tokenizer=str(ok))
    assert (st.stop_tokens, st.missing_eos) == (["<|im_end|>"], 0)

    # Qwen base-style: eos is <|endoftext|> but the template ends turns with <|im_end|>.
    base = _tokenizer(tmp_path / "base", eos="<|endoftext|>")
    st = check_tokens(src, tokenizer=str(base))
    assert st.missing_eos == 1
    assert any("does not learn to stop" in h for h in st.hints())

    # ...unless generation_config.json lists <|im_end|> as a stop token (Gemma-style).
    fixed = _tokenizer(tmp_path / "fixed", eos="<|endoftext|>", generation_eos=[3, 2])
    st = check_tokens(src, tokenizer=str(fixed))
    assert (st.stop_tokens, st.missing_eos) == (["<|endoftext|>", "<|im_end|>"], 0)

    assert check_tokens(src, tokenizer=str(ok), chat_template=NO_END).missing_eos == 1


def test_reasoning_dropped(tmp_path: Path, tok_dir: Path) -> None:
    multi = {
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "first trace"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "reasoning_content": "second trace"},
        ]
    }
    src = _write(tmp_path / "a.jsonl", [multi])
    st = check_tokens(src, tokenizer=str(tok_dir))
    assert (st.reasoning_dropped, st.reasoning_dropped_final) == (1, 0)
    assert any("--reasoning-turns last" in h for h in st.hints())

    # A gpt-oss style file (thinking) against a template that reads reasoning_content.
    thinking = _write(
        tmp_path / "t.jsonl", [_msgs(("user", "q"), ("assistant", "a"), thinking="t r")]
    )
    st = check_tokens(thinking, tokenizer=str(tok_dir))
    assert (st.reasoning_dropped, st.reasoning_dropped_final) == (1, 1)
    assert any("--reasoning thinking" in h for h in st.hints())

    inline = _write(
        tmp_path / "i.jsonl", [_msgs(("user", "q"), ("assistant", "<think>t</think> a"))]
    )
    assert check_tokens(inline, tokenizer=str(tok_dir)).reasoning_dropped == 0


def test_null_tool_content_is_rendered_as_stored(tmp_path: Path, tok_dir: Path) -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    row = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "content": "x", "tool_call_id": "c1"},
            {"role": "assistant", "content": "done"},
        ]
    }
    src = _write(tmp_path / "n.jsonl", [row])
    st = check_tokens(src, tokenizer=str(tok_dir), chat_template=CONCAT)
    assert sum(st.template_errors.values()) == 1
    assert any("--tool-content empty" in h for h in st.hints())
    row["messages"][1]["content"] = ""
    fixed = _write(tmp_path / "e.jsonl", [row])
    assert not check_tokens(fixed, tokenizer=str(tok_dir), chat_template=CONCAT).template_errors


def test_cli_prints_hints_without_failing(tmp_path: Path, tok_dir: Path, capsys) -> None:
    src = _write(tmp_path / "a.jsonl", [_msgs(("user", "hi"), ("assistant", "yo"))])
    main(["tokens", "-i", str(src), "--tokenizer", str(tok_dir)])  # exit 0: warnings only
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["generation_tags"] is False and report["hints"]
    assert "hint: the template has no {% generation %}" in captured.err


def test_hints_name_the_tool_argument_encoding_a_template_wants() -> None:
    from convmerge.tokens import TokenStats

    wants_string = TokenStats(
        template_errors={'TypeError: can only concatenate str (not "dict") to str': 3}
    )
    assert any("--tool-arguments string" in h for h in wants_string.hints())
    wants_object = TokenStats(
        template_errors={"UndefinedError: 'str object' has no attribute 'items'": 1}
    )
    assert any("--tool-arguments object" in h for h in wants_object.hints())
