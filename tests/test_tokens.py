"""convmerge tokens: lengths, chat-template failures, and filtering.

Uses a tiny word-level tokenizer built offline (one token per word or
punctuation mark), so lengths are easy to predict and no network is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from convmerge.cli import main  # noqa: E402
from convmerge.tokens import TokenStats, check_tokens  # noqa: E402

CHATML = (
    "{% for m in messages %}<|im_start|> {{ m.role }} {{ m.content or '' }}"
    "{% if m.tool_calls %}{% for c in m.tool_calls %} call {{ c.function.name }}"
    "{% endfor %}{% endif %} <|im_end|> {% endfor %}"
)
STRICT = (
    "{% for m in messages %}{% if (m.role == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}{{ m.content }} {% endfor %}"
)


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "<|im_start|>": 1, "<|im_end|>": 2}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    fast.chat_template = CHATML
    path = tmp_path_factory.mktemp("tok")
    fast.save_pretrained(path)
    return path


def M(*turns: tuple[str, str]) -> list[dict]:
    return [{"role": r, "content": c} for r, c in turns]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_lengths_and_report(tmp_path: Path, tokenizer_dir: Path) -> None:
    # "<|im_start|> user a b c <|im_end|>" = 6 tokens per turn with 3 words.
    rows = [
        {"messages": M(("user", "a b c"), ("assistant", "d e f"))},  # 12
        {"messages": M(("user", "a"), ("assistant", "b"))},  # 8
    ]
    stats = check_tokens(_write(tmp_path / "in.jsonl", rows), tokenizer=str(tokenizer_dir))
    report = stats.to_report()
    assert report["measured"] == 2 and report["template_errors"] == {}
    assert report["tokens"]["min"] == 8 and report["tokens"]["max"] == 12
    assert report["tokens"]["total"] == 20 and report["histogram"] == {"<256": 2}


def test_filter_by_max_tokens_keeps_rows_verbatim(tmp_path: Path, tokenizer_dir: Path) -> None:
    rows = [{"messages": M(("user", "w " * n), ("assistant", "ok"))} for n in (1, 5, 20)]
    src = _write(tmp_path / "in.jsonl", rows)
    stats = TokenStats()
    check_tokens(src, tokenizer=str(tokenizer_dir), max_tokens=12, output=tmp_path / "kept.jsonl",
                 rejects=tmp_path / "rej.jsonl", stats=stats)  # fmt: skip
    lines = src.read_text().splitlines()
    assert (tmp_path / "kept.jsonl").read_text().splitlines() == lines[:2]
    assert (tmp_path / "rej.jsonl").read_text().splitlines() == lines[2:]
    assert (stats.kept, stats.rejected, stats.over_limit) == (2, 1, 1)


def test_template_failures_are_grouped_and_rejected(tmp_path: Path, tokenizer_dir: Path) -> None:
    rows = [
        {"messages": M(("user", "a"), ("assistant", "b"))},
        {"messages": M(("system", "s"), ("user", "a"), ("assistant", "b"))},
        {"messages": M(("system", "s"), ("user", "c"), ("assistant", "d"))},
    ]
    src = _write(tmp_path / "in.jsonl", rows)
    stats = check_tokens(src, tokenizer=str(tokenizer_dir), chat_template=STRICT,
                         output=tmp_path / "ok.jsonl")  # fmt: skip
    (reason,) = stats.template_errors
    assert "roles must alternate" in reason and stats.template_errors[reason] == 2
    assert stats.error_lines[reason] == [2, 3]
    assert stats.kept == 1 and stats.rejected == 2


def test_preference_rows_count_their_longer_side(tmp_path: Path, tokenizer_dir: Path) -> None:
    row = {"prompt": M(("user", "q")), "chosen": M(("assistant", "short")),
           "rejected": M(("assistant", "a much longer answer here"))}  # fmt: skip
    stats = check_tokens(_write(tmp_path / "p.jsonl", [row]), tokenizer=str(tokenizer_dir))
    assert list(stats.lengths) == [4 + 8]  # prompt turn 4 + rejected turn 8 (chosen: 5)


def test_any_layout_and_tool_calls(tmp_path: Path, tokenizer_dir: Path) -> None:
    call = {"type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}
    rows = [
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]},
        {"instruction": "q", "output": "a"},
        {"messages": [{"role": "user", "content": "go"},
                      {"role": "assistant", "content": None, "tool_calls": [call]}]},
        {"unrelated": 1},
    ]  # fmt: skip
    stats = check_tokens(_write(tmp_path / "mix.jsonl", rows), tokenizer=str(tokenizer_dir))
    assert stats.measured == 3 and stats.unreadable == 1
    assert list(stats.lengths)[2] == 4 + 5  # "<|im_start|> assistant call f <|im_end|>"


def test_tokenizer_without_template(tmp_path: Path, tokenizer_dir: Path) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    tok.chat_template = None
    src = _write(tmp_path / "in.jsonl", [{"messages": M(("user", "a"), ("assistant", "b"))}])
    with pytest.raises(ValueError, match="no chat template"):
        check_tokens(src, tokenizer=tok)


def test_cli(tmp_path: Path, tokenizer_dir: Path, capsys) -> None:
    rows = [{"messages": M(("user", "w " * n), ("assistant", "ok"))} for n in (1, 30)]
    src = _write(tmp_path / "in.jsonl", rows)
    with pytest.raises(SystemExit) as exc:  # check mode: over-limit rows fail
        main(["tokens", "-i", str(src), "--tokenizer", str(tokenizer_dir), "--max-tokens", "20"])
    assert exc.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["over_limit"] == 1 and report["tokenizer"] == str(tokenizer_dir)
    main(["tokens", "-i", str(src), "--tokenizer", str(tokenizer_dir), "--max-tokens", "20",
          "-o", str(tmp_path / "ok.jsonl")])  # fmt: skip
    assert "kept 1" in capsys.readouterr().err
    strict = tmp_path / "strict.jinja"
    strict.write_text(STRICT)
    main(["tokens", "-i", str(src), "--tokenizer", str(tokenizer_dir), "--chat-template",
          str(strict)])  # fmt: skip
    with pytest.raises(SystemExit) as exc:
        main(["tokens", "-i", str(src), "--tokenizer", str(tmp_path / "no-such-tokenizer")])
    assert exc.value.code == 2
