"""Independent regressions for the six issue families found in convmerge 1.6.4.

Run with: python -m pytest test_remaining_issues.py -q
All destructive inputs are generated under pytest's temporary directory.
Failures document unmet requirements; this does not patch the package.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def chat(q: str = "Question", a: str = "Answer") -> dict:
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def rows(path: Path, records: list[dict]) -> Path:
    return write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cli(root: Path, *args: object) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return subprocess.run(
        [sys.executable, "-m", "convmerge", *map(str, args)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def recipe(root: Path, overrides: dict | None = None) -> tuple[Path, Path]:
    source = rows(
        root / "source.jsonl", [{"instruction": "Question", "output": "Answer", "id": "original"}]
    )
    cfg = {
        "version": 1,
        "workdir": "build",
        "output": "train.jsonl",
        "sources": {"a": {"path": "source.jsonl", "normalize": False, "convert": {"from": "auto"}}},
    }
    cfg.update(overrides or {})
    return source, write(root / "recipe.json", json.dumps(cfg))


@pytest.mark.parametrize(
    "overrides",
    [
        {"output": "source.jsonl"},
        {"report": "train.jsonl"},
        {"lock": "source.jsonl"},
        {"report": "source.jsonl"},
        {"output": "recipe.json"},
        {"lock": "train.jsonl"},
    ],
    ids=[
        "output-source",
        "report-output",
        "lock-source",
        "report-source",
        "output-recipe",
        "lock-output",
    ],
)
def test_recipe_rejects_aliases_before_writing(tmp_path: Path, overrides: dict) -> None:
    source, cfg = recipe(tmp_path, overrides)
    before = digest(source), digest(cfg)
    proc = cli(tmp_path, "run", cfg)
    assert proc.returncode == 2, f"unsafe recipe reported exit {proc.returncode}: {proc.stderr}"
    assert (digest(source), digest(cfg)) == before
    assert not (tmp_path / "build").exists()


def test_recipe_control_and_cache(tmp_path: Path) -> None:
    source, cfg = recipe(tmp_path)
    before = digest(source), digest(cfg)
    first = cli(tmp_path, "run", cfg)
    assert first.returncode == 0, first.stderr
    out = tmp_path / "train.jsonl"
    assert json.loads(out.read_text()) == chat()
    previous = digest(out)
    second = cli(tmp_path, "run", cfg)
    assert second.returncode == 0, second.stderr
    assert digest(out) == previous
    assert (digest(source), digest(cfg)) == before
    assert "[run]" not in second.stderr


def test_normalize_valid_long_first_line(tmp_path: Path) -> None:
    from convmerge.io import iter_jsonl

    records = [chat("Long question", "x" * 70000), chat()]
    src = rows(tmp_path / "input.jsonl", records)
    assert len(list(iter_jsonl(src))) == 2
    out = tmp_path / "out.jsonl"
    proc = cli(tmp_path, "normalize", "-i", src, "-o", out)
    assert proc.returncode == 0, proc.stderr
    assert [json.loads(x) for x in out.read_text().splitlines()] == records


def test_normalize_does_not_mistake_blank_prefix_for_empty_file(tmp_path: Path) -> None:
    from convmerge.io import iter_jsonl

    src = write(tmp_path / "input.jsonl", "\n" * 70000 + json.dumps(chat()) + "\n")
    assert len(list(iter_jsonl(src))) == 1
    out = tmp_path / "out.jsonl"
    proc = cli(tmp_path, "normalize", "-i", src, "-o", out)
    assert proc.returncode == 0, proc.stderr
    assert len(list(iter_jsonl(out))) == 1, "normalization silently removed the valid record"


def test_normalize_long_later_row_control(tmp_path: Path) -> None:
    records = [chat(), chat("Long question", "x" * 70000)]
    src = rows(tmp_path / "input.jsonl", records)
    out = tmp_path / "out.jsonl"
    proc = cli(tmp_path, "normalize", "-i", src, "-o", out)
    assert proc.returncode == 0, proc.stderr
    assert [json.loads(x) for x in out.read_text().splitlines()] == records


def test_normalize_actual_empty_control(tmp_path: Path) -> None:
    src = write(tmp_path / "input.jsonl", "")
    out = tmp_path / "out.jsonl"
    proc = cli(tmp_path, "normalize", "-i", src, "-o", out)
    assert proc.returncode == 0, proc.stderr
    assert out.read_bytes() == b""


@pytest.fixture
def tokenizer_dir(tmp_path: Path):
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "</s>": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="</s>")
    tok.chat_template = (
        "{% for m in messages %}{{ m.content or '' }} "
        "{% for c in m.tool_calls|default([]) %}"
        "{% for k,v in c.function.arguments.items() %}{{ k }} {{ v }} {% endfor %}"
        "{% endfor %} </s> {% endfor %}"
    )
    folder = tmp_path / "tokenizer"
    tok.save_pretrained(folder)
    return tok, folder


def tool_row(mixed: bool) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "Get weather for both cities."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "one",
                        "type": "function",
                        "function": {"name": "weather", "arguments": {"city": "Seoul"}},
                    },
                    {
                        "id": "two",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city":"Busan"}' if mixed else {"city": "Busan"},
                        },
                    },
                ],
            },
        ]
    }


def test_tokens_does_not_keep_row_its_own_template_cannot_render(
    tmp_path: Path, tokenizer_dir
) -> None:
    tok, folder = tokenizer_dir
    record = tool_row(True)
    with pytest.raises(Exception, match="items"):
        tok.apply_chat_template(record["messages"], tokenize=False)
    src = rows(tmp_path / "input.jsonl", [record])
    out = tmp_path / "kept.jsonl"
    proc = cli(tmp_path, "tokens", "-i", src, "--tokenizer", folder, "-o", out)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["kept"] == 0, f"invalid original row was kept unchanged: {report}"
    assert report["template_errors"]
    assert out.read_bytes() == b""


def test_tokens_homogeneous_arguments_control(tmp_path: Path, tokenizer_dir) -> None:
    tok, folder = tokenizer_dir
    record = tool_row(False)
    assert tok.apply_chat_template(record["messages"], tokenize=False)
    src = rows(tmp_path / "input.jsonl", [record])
    out = tmp_path / "kept.jsonl"
    proc = cli(tmp_path, "tokens", "-i", src, "--tokenizer", folder, "-o", out)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["kept"] == 1
    assert out.read_bytes() == src.read_bytes()


def test_filter_protects_rules_file(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    src = rows(tmp_path / "input.jsonl", [chat()])
    rules = write(tmp_path / "rules.yaml", "refusal: []\n")
    before = digest(rules)
    proc = cli(tmp_path, "filter", "-i", src, "--rules-file", rules, "-o", rules)
    assert proc.returncode == 2, proc.stderr
    assert digest(rules) == before


@pytest.mark.parametrize("kind", ["template", "tokenizer-config"])
def test_tokens_protects_configuration_files(tmp_path: Path, tokenizer_dir, kind: str) -> None:
    tok, folder = tokenizer_dir
    src = rows(tmp_path / "input.jsonl", [chat()])
    extra = []
    if kind == "template":
        config = write(tmp_path / "template.jinja", tok.chat_template)
        extra = ["--chat-template", config]
    else:
        config = folder / "tokenizer_config.json"
    before = digest(config)
    proc = cli(tmp_path, "tokens", "-i", src, "--tokenizer", folder, "-o", config, *extra)
    assert proc.returncode == 2, proc.stderr
    assert digest(config) == before


def test_llamafactory_info_protects_input(tmp_path: Path) -> None:
    src = rows(
        tmp_path / "data.jsonl",
        [{"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}],
    )
    before = digest(src)
    proc = cli(tmp_path, "llamafactory-info", "-i", src, "--info", src, "--name", "dataset")
    assert proc.returncode == 2, proc.stderr
    assert digest(src) == before


@pytest.mark.parametrize(
    "payload", ["[]", '{"version":1,"steps":[],"files":{}}'], ids=["root-list", "steps-list"]
)
def test_malformed_lock_is_handled_without_traceback(tmp_path: Path, payload: str) -> None:
    source, config = recipe(tmp_path)
    before = digest(source)
    write(tmp_path / "recipe.lock.json", payload)
    proc = cli(tmp_path, "run", config)
    assert "Traceback" not in proc.stderr, proc.stderr
    assert digest(source) == before
    if proc.returncode == 0:
        assert json.loads((tmp_path / "train.jsonl").read_text()) == chat()
    else:
        assert "lock" in proc.stderr.lower()


def test_mix_scaling_finite_weights_preserves_distribution(tmp_path: Path) -> None:
    from convmerge.mix import MixSource, mix_files

    a = rows(tmp_path / "a.jsonl", [chat(f"A {i}", "Answer") for i in range(3)])
    b = rows(tmp_path / "b.jsonl", [chat(f"B {i}", "Answer") for i in range(3)])
    control = tmp_path / "control.jsonl"
    large = tmp_path / "large.jsonl"
    normal = mix_files([MixSource(a, 1.0), MixSource(b, 1.0)], control, total=4)
    assert normal.total_written == 4
    scaled = mix_files([MixSource(a, 1e308), MixSource(b, 1e308)], large, total=4)
    assert scaled.total_written == 4
    assert [s.weight for s in scaled.sources] == [0.5, 0.5]
    assert large.read_bytes() == control.read_bytes()


def test_previously_fixed_normalize_failure_control(tmp_path: Path) -> None:
    src = write(tmp_path / "bad.jsonl", '{"a":1}\n{broken\n')
    out = rows(tmp_path / "out.jsonl", [chat("previous", "output")])
    before = digest(out)
    proc = cli(tmp_path, "normalize", "-i", src, "-o", out)
    assert proc.returncode == 1
    assert digest(out) == before


def test_previously_fixed_convert_failure_control(tmp_path: Path) -> None:
    src = rows(
        tmp_path / "bad.jsonl", [chat(), {"messages": [{"role": "user", "content": "no answer"}]}]
    )
    out = rows(tmp_path / "out.jsonl", [chat("previous", "output")])
    before = digest(out)
    proc = cli(tmp_path, "convert", "-i", src, "-o", out, "--from", "auto", "--on-invalid", "fail")
    assert proc.returncode == 1
    assert digest(out) == before
