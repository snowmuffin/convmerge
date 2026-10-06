"""Expanded 1.6.5 boundaries: originals, configuration, raw renderings and cache schema."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.io import SamePathError, iter_jsonl
from convmerge.normalize.jsonl import detect_jsonl_shape, normalize_to_jsonl
from convmerge.recipe import RecipeError, load_lock, parse_recipe, plan, run


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _chat(answer="answer"):
    return {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": answer},
        ]
    }


def _recipe(root, overrides=None):
    _write(root / "source.jsonl", json.dumps(_chat()) + "\n")
    spec = {
        "output": "train.jsonl",
        "workdir": "build",
        "sources": {"a": {"path": "source.jsonl", "normalize": False, "convert": {"from": "auto"}}},
    }
    spec.update(overrides or {})
    _write(root / "recipe.json", json.dumps(spec))
    return parse_recipe(spec, path=root / "recipe.json")


@pytest.mark.parametrize(
    "prefix",
    ["", "\ufeff", "\n" * 70000, " " * 70000],
    ids=["plain", "bom", "blank-lines", "spaces"],
)
@pytest.mark.parametrize("ending", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_complete_large_first_record(tmp_path, prefix, ending):
    records = [_chat("한글🙂" * 24000), _chat()]
    text = prefix + ending.join(json.dumps(r, ensure_ascii=False) for r in records) + ending
    src = _write(tmp_path / "input.jsonl", text)
    dst = tmp_path / "out.jsonl"
    assert normalize_to_jsonl(src, dst) == 2
    assert [line.value for line in iter_jsonl(dst)] == records


@pytest.mark.parametrize("kind", ["single", "array", "array-lines", "concatenated", "blank"])
def test_large_shape_detection(tmp_path, kind):
    record = {"text": "x" * 70000}
    encoded = json.dumps(record)
    data, expected, shape = {
        "single": (encoded, [record], "single_line"),
        "array": ("[" + encoded + "]", [record], "json_array"),
        "array-lines": (
            f"[{encoded}]\n[{encoded}]\n",
            [{"conversation": [record]}] * 2,
            "jsonl_of_arrays",
        ),
        "concatenated": (encoded + encoded, [record, record], "single_line"),
        "blank": ("\n " * 70000, [], "empty"),
    }[kind]
    src = _write(tmp_path / "input.json", data)
    dst = tmp_path / "out.jsonl"
    assert detect_jsonl_shape(src) == shape
    assert normalize_to_jsonl(src, dst) == len(expected)
    assert [line.value for line in iter_jsonl(dst)] == expected


@pytest.mark.parametrize("slot", ["output", "lock", "report"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_recipe_alias_preflight_in_run_and_plan(tmp_path, slot, kind):
    recipe = _recipe(tmp_path, {slot: "alias.json"})
    source = tmp_path / "source.jsonl"
    alias = tmp_path / "alias.json"
    try:
        if kind == "symlink":
            alias.symlink_to(source)
        else:
            os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"filesystem does not support {kind}: {exc}")
    before = source.read_bytes()
    for operation in (plan, run):
        with pytest.raises(SamePathError):
            operation(recipe)
    assert source.read_bytes() == before
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize(
    "variant",
    ["output-in-source", "work-in-source", "output-parent", "metadata-in-stage", "duplicate-stage"],
)
def test_recipe_tree_relationships(tmp_path, variant):
    _write(tmp_path / "raw" / "data.jsonl", json.dumps(_chat()) + "\n")
    overrides = {
        "output-in-source": {"output": "raw/train.jsonl"},
        "work-in-source": {"workdir": "raw/build"},
        "output-parent": {"output": "raw"},
        "metadata-in-stage": {"report": "build/a/jsonl/report.json"},
        "duplicate-stage": {"output": "build/a/converted.jsonl", "dedupe": True},
    }[variant]
    recipe = _recipe(
        tmp_path, {"sources": {"a": {"path": "raw", "convert": {"from": "auto"}}}, **overrides}
    )
    before = (tmp_path / "raw/data.jsonl").read_bytes()
    with pytest.raises(SamePathError):
        run(recipe)
    assert (tmp_path / "raw/data.jsonl").read_bytes() == before
    assert not (tmp_path / "build").exists()


def test_recipe_protects_manifest_before_network(tmp_path):
    pytest.importorskip("yaml")
    manifest = _write(
        tmp_path / "manifest.yaml", "datasets:\n  - name: remote\n    hf: unused/test-fixture\n"
    )
    recipe = _recipe(
        tmp_path,
        {
            "output": "manifest.yaml",
            "sources": {
                "a": {
                    "fetch": {"manifest": "manifest.yaml", "name": "remote"},
                    "convert": {"from": "auto"},
                }
            },
        },
    )
    before = manifest.read_bytes()
    with pytest.raises(SamePathError):
        run(recipe)
    assert manifest.read_bytes() == before
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize(
    "entry",
    [
        None,
        [],
        {"inputs": []},
        {"options": None},
        {"report": []},
        {"report": {"stats": []}},
        {"inputs": {"data": []}},
        {"output": 42},
    ],
    ids=["null", "list", "inputs", "options", "report", "stats", "digest-value", "output"],
)
def test_invalid_lock_step_schema_is_preserved(tmp_path, entry):
    recipe = _recipe(tmp_path)
    lock = _write(recipe.lock_path, json.dumps({"version": 1, "steps": {"a.convert": entry}}))
    before = lock.read_bytes()
    for operation in (load_lock,):
        with pytest.raises(RecipeError, match="lock"):
            operation(lock)
    for operation in (plan, run):
        with pytest.raises(RecipeError, match="lock"):
            operation(recipe)
    assert lock.read_bytes() == before
    assert not recipe.workdir.exists()


@pytest.mark.parametrize("value", [None, [], [1], [1, 2], [1, 2, {}], [True, 2, "abc"]])
def test_invalid_lock_digest_cache(tmp_path, value):
    lock = _write(tmp_path / "lock.json", json.dumps({"version": 1, "files": {"input": value}}))
    with pytest.raises(RecipeError, match="files.input"):
        load_lock(lock)


@pytest.mark.parametrize("mode", [[], ["--plan"], ["--frozen"]], ids=["run", "plan", "frozen"])
def test_bad_lock_cli_message_and_exit(tmp_path, capsys, mode):
    recipe = _recipe(tmp_path)
    lock = _write(recipe.lock_path, "{broken")
    with pytest.raises(SystemExit) as exc:
        main(["run", str(recipe.path), *mode])
    assert exc.value.code == 2
    assert "lock" in capsys.readouterr().err
    assert lock.read_text() == "{broken"
    assert not recipe.workdir.exists()


@pytest.fixture
def tokenizer(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "</s>": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="</s>")
    tok.chat_template = "{% for m in messages %}{{ m.content or '' }} </s> {% endfor %}"
    folder = tmp_path / "tokenizer"
    tok.save_pretrained(folder)
    return tok, folder


@pytest.mark.parametrize(
    "payload",
    ["", "\n " * 4, "{bad\n", '{"unknown":1}\n'],
    ids=["empty", "blank", "invalid-json", "unknown-record"],
)
def test_tokens_check_mode_never_succeeds_without_measuring(tmp_path, capsys, tokenizer, payload):
    _, folder = tokenizer
    src = _write(tmp_path / "in.jsonl", payload)
    with pytest.raises(SystemExit) as exc:
        main(["tokens", "-i", str(src), "--tokenizer", str(folder)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["measured"] == 0
    assert "no rows could be measured" in captured.err


def test_tokens_check_rejects_partial_validation_but_filtering_can_continue(
    tmp_path, capsys, tokenizer
):
    _, folder = tokenizer
    src = _write(tmp_path / "in.jsonl", json.dumps(_chat()) + "\n{bad\n")
    with pytest.raises(SystemExit) as exc:
        main(["tokens", "-i", str(src), "--tokenizer", str(folder)])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["measured"] == 1
    out = tmp_path / "kept.jsonl"
    main(["tokens", "-i", str(src), "--tokenizer", str(folder), "-o", str(out)])
    assert json.loads(capsys.readouterr().out)["kept"] == 1


@pytest.mark.parametrize("field", ["output", "rejects"])
def test_direct_tokens_protects_local_tokenizer(tmp_path, tokenizer, field):
    from convmerge.tokens import check_tokens

    _, folder = tokenizer
    src = _write(tmp_path / "in.jsonl", json.dumps(_chat()) + "\n")
    config = folder / "tokenizer_config.json"
    before = config.read_bytes()
    with pytest.raises(SamePathError):
        check_tokens(src, tokenizer=str(folder), **{field: config})
    assert config.read_bytes() == before


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("preference", [False, True])
def test_raw_arguments_each_call_and_both_dpo_sides(tmp_path, tokenizer, reverse, preference):
    from convmerge.tokens import check_tokens

    tok, _ = tokenizer
    tok.chat_template = (
        "{% for m in messages %}{{ m.content or '' }} "
        "{% for c in m.tool_calls|default([]) %}"
        "{% for k,v in c.function.arguments.items() %}{{ k }} {{ v }} {% endfor %}"
        "{% endfor %} </s> {% endfor %}"
    )
    calls = [
        {
            "id": str(i),
            "type": "function",
            "function": {
                "name": "f",
                "arguments": {"city": "Seoul"} if i == 0 else '{"city":"Busan"}',
            },
        }
        for i in range(2)
    ]
    if reverse:
        calls.reverse()
    answer = {"role": "assistant", "content": "", "tool_calls": calls}
    row = _chat()
    if preference:
        row = {
            "prompt": row["messages"][:1],
            "chosen": [{"role": "assistant", "content": "ordinary response"}],
            "rejected": [answer],
        }
    else:
        row["messages"][-1] = answer
    src = _write(tmp_path / "in.jsonl", json.dumps(row) + "\n")
    out = tmp_path / "out.jsonl"
    st = check_tokens(src, tokenizer=tok, output=out)
    assert st.kept == 0 and st.rejected == 1
    assert sum(st.template_errors.values()) == 1
    assert out.read_bytes() == b""


@pytest.mark.parametrize("variant", ["missing", "null", "empty", "reasoning-flags"])
def test_native_messages_are_rendered_without_reconstruction(tmp_path, tokenizer, variant):
    from convmerge.tokens import check_tokens

    tok, _ = tokenizer
    tok.chat_template = (
        "{% for m in messages %}{{ m.role }}:{{ m.content|default('ABSENT') }} "
        "{{ m.train|default('DEFAULT') }} {{ m.thinking|default('') }} "
        "{{ m.reasoning_content|default('') }} </s> {% endfor %}"
    )
    row = _chat()
    if variant != "reasoning-flags":
        row["messages"][-1]["tool_calls"] = [
            {"type": "function", "function": {"name": "f", "arguments": {}}}
        ]
        if variant == "missing":
            del row["messages"][-1]["content"]
        else:
            row["messages"][-1]["content"] = None if variant == "null" else ""
    else:
        row["messages"][-1].update(thinking="keep this trace", train=False)
        row["messages"] += copy.deepcopy(_chat()["messages"])
        row["messages"][-1]["reasoning_content"] = "keep different field"
    direct = tok.apply_chat_template(row["messages"], tokenize=False)
    expected = len(tok(direct, add_special_tokens=False)["input_ids"])
    src = _write(tmp_path / "in.jsonl", json.dumps(row) + "\n")
    st = check_tokens(src, tokenizer=tok)
    assert list(st.lengths) == [expected]
    assert not st.template_errors


@pytest.mark.parametrize("sampler", ["v1", "v2"])
def test_large_finite_weights_match_scaled_equivalent(tmp_path, sampler):
    from convmerge.mix import MixSource, mix_files

    sources = [
        _write(
            tmp_path / f"{i}.jsonl",
            "".join(json.dumps(_chat(f"source {i} row {j}")) + "\n" for j in range(10)),
        )
        for i in range(2)
    ]
    files = []
    for label, weights in [("control", [1, 1]), ("large", [1e308, 1e308])]:
        output = tmp_path / f"{label}.jsonl"
        result = mix_files(
            [MixSource(path, weight) for path, weight in zip(sources, weights)],
            output,
            total=8,
            sampler=sampler,
        )
        assert result.total_written == 8
        assert [s.weight for s in result.sources] == [0.5, 0.5]
        files.append(output.read_bytes())
    assert files[0] == files[1]
