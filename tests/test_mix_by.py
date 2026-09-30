"""mix --by chars / tokens and the reasoning report (1.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli.mix import mix_summary
from convmerge.mix import MixSource, load_mix_config, mix_files


def _rows(path: Path, n: int, length: int, *, think: bool = False) -> Path:
    answer = ("<think>hm</think>" if think else "") + "x" * length
    rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": answer}]} for i in range(n)]  # fmt: skip
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _chars(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        answer = json.loads(line)["messages"][1]["content"]
        key = "long" if len(answer) > 200 else "short"
        out[key] = out.get(key, 0) + sum(len(m["content"]) for m in json.loads(line)["messages"])
    return out


def test_by_chars_splits_the_text_by_weight(tmp_path: Path) -> None:
    short = _rows(tmp_path / "short.jsonl", 2000, 100)
    long = _rows(tmp_path / "long.jsonl", 2000, 400, think=True)
    sources = [MixSource(short, 0.7), MixSource(long, 0.3)]
    by_rows = mix_files(sources, tmp_path / "rows.jsonl", total=1000)
    assert [s.written for s in by_rows.sources] == [700, 300]
    result = mix_files(sources, tmp_path / "chars.jsonl", total=1000, by="chars")
    assert result.by == "chars" and result.total_written == 1000
    chars = _chars(tmp_path / "chars.jsonl")
    assert chars["short"] / (chars["short"] + chars["long"]) == pytest.approx(0.7, abs=0.01)
    assert result.sources[1].reasoning == 2000 and result.sources[0].reasoning == 0
    assert result.sources[0].mean_units == pytest.approx(104.4, abs=0.1)
    lines = mix_summary(result, oversample=False)
    assert "chars=70.0%" in lines[0] and "reasoning=100%" in lines[1]
    assert lines[-1].startswith("rows with a reasoning trace: about ")
    # the sidecar records it
    from convmerge.mix import write_mix_recipe

    sidecar = json.loads(write_mix_recipe(result).read_text())
    assert sidecar["by"] == "chars" and sidecar["sources"][1]["reasoning"] == 2000


class _Tok:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return text.split()


def test_by_tokens_uses_the_tokenizer(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps({"text": "w " * 10}) + "\n" for _ in range(500)))
    b.write_text("".join(json.dumps({"text": "w " * 40}) + "\n" for _ in range(500)))
    result = mix_files([MixSource(a, 1), MixSource(b, 1)], tmp_path / "o.jsonl", total=500,
                       by="tokens", tokenizer=_Tok())  # fmt: skip
    assert [s.written for s in result.sources] == [400, 100]
    assert [s.units for s in result.sources] == [5000, 20000]


def test_by_sample_estimates_from_a_seeded_sample(tmp_path: Path) -> None:
    short = _rows(tmp_path / "short.jsonl", 3000, 100)
    long = _rows(tmp_path / "long.jsonl", 3000, 400)
    sources = [MixSource(short, 0.7), MixSource(long, 0.3)]
    exact = mix_files(sources, tmp_path / "e.jsonl", total=1000, by="chars")
    est = mix_files(sources, tmp_path / "s.jsonl", total=1000, by="chars", by_sample=200)
    mix_files(sources, tmp_path / "t.jsonl", total=1000, by="chars", by_sample=200)
    assert [s.measured for s in exact.sources] == [3000, 3000]
    assert [s.measured for s in est.sources] == [200, 200] and est.by_sample == 200
    for e, s in zip(exact.sources, est.sources):
        assert s.units == pytest.approx(e.units, rel=0.01)
        assert abs(s.written - e.written) <= 5
    assert (tmp_path / "s.jsonl").read_text() == (tmp_path / "t.jsonl").read_text()
    lines = mix_summary(est, oversample=False)
    assert lines[-1] == "chars per row estimated from 200 random rows of 2 source(s)"
    # a sample as large as the source measures every row: no estimate
    whole = mix_files(sources, tmp_path / "w.jsonl", total=1000, by="chars", by_sample=5000)
    assert [s.units for s in whole.sources] == [s.units for s in exact.sources]
    assert not mix_summary(whole, oversample=False)[-1].startswith("chars per row")
    from convmerge.mix import write_mix_recipe

    sidecar = json.loads(write_mix_recipe(est).read_text())
    assert sidecar["by_sample"] == 200 and sidecar["sources"][0]["measured"] == 200


@pytest.mark.parametrize(
    "kwargs, message",
    [({"by_sample": 5}, "needs by='chars'"), ({"by": "chars", "by_sample": 0}, "positive")],
)
def test_by_sample_errors(tmp_path: Path, kwargs, message) -> None:
    src = _rows(tmp_path / "s.jsonl", 3, 5)
    with pytest.raises(ValueError, match=message):
        mix_files([MixSource(src, 1)], tmp_path / "o.jsonl", total=2, **kwargs)


def test_by_sample_cli_config_and_recipe(tmp_path: Path, capsys) -> None:
    from convmerge.cli import main
    from convmerge.recipe import RecipeError
    from convmerge.recipe.schema import parse_recipe

    src = _rows(tmp_path / "a.jsonl", 300, 50)
    main(["mix", "-i", f"{src}:1", "-o", str(tmp_path / "o.jsonl"), "-n", "10", "--by", "chars",
          "--by-sample", "20", "--no-recipe"])  # fmt: skip
    assert "estimated from 20 random rows" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["mix", "-i", f"{src}:1", "-o", str(tmp_path / "p.jsonl"), "--by-sample", "0"])
    assert e.value.code == 2
    cfg = tmp_path / "mix.json"
    cfg.write_text(json.dumps({"sources": [{"path": str(src), "weight": 1}], "total": 5,
                               "by": "chars", "by_sample": 7}))  # fmt: skip
    assert load_mix_config(cfg)[1]["by_sample"] == 7
    base = {"version": 1, "output": "o.jsonl",
            "sources": {"a": {"path": "a.jsonl", "convert": {"from": "auto"}}}}  # fmt: skip
    mix = {"total": 10, "by": "chars", "by_sample": 50}
    assert parse_recipe({**base, "mix": mix}, path=tmp_path / "r.yaml").mix.by_sample == 50
    with pytest.raises(RecipeError, match="needs mix.by chars or tokens"):
        parse_recipe({**base, "mix": {"total": 10, "by_sample": 5}}, path=tmp_path / "r.yaml")
    with pytest.raises(RecipeError, match="positive integer"):
        parse_recipe({**base, "mix": {**mix, "by_sample": True}}, path=tmp_path / "r.yaml")


def test_reasoning_count_sees_keys_and_escaped_tags(tmp_path: Path) -> None:
    rows = [
        {"messages": [{"role": "assistant", "content": "a", "reasoning_content": "r"}]},
        {"conversations": [{"from": "gpt", "value": "a", "thinking": "t"}]},
        {"messages": [{"role": "assistant", "content": "a", "reasoning": "  "}]},  # blank
        {"messages": [{"role": "assistant", "content": "<think>t</think>a"}]},
        {"messages": [{"role": "assistant", "content": "a <b>"}]},
        {"output": "<think>t</think>a"},
        {"messages": ["reasoning", 1]},
    ]
    text = "".join(json.dumps(r) + "\n" for r in rows)
    text += '{"messages": [{"role": "assistant", "content": "\\u003Cthink>t</think>a"}]}\n'
    text += '{"messages": [{"role": "assistant", "content": "a", "\\u0074hinking": "t"}]}\n'
    src = tmp_path / "a.jsonl"
    src.write_text(text, encoding="utf-8")
    for sampler in ("v2", "v1"):
        result = mix_files([MixSource(src, 1.0)], tmp_path / "o.jsonl", total=9, sampler=sampler)
        assert result.sources[0].reasoning == 6


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"by": "chars"}, "needs a total"),
        ({"by": "chars", "total": 5, "sampler": "v1"}, "v2 sampler"),
        ({"by": "tokens", "total": 5}, "needs a tokenizer"),
        ({"by": "bytes", "total": 5}, "by must be"),
    ],
)
def test_by_errors(tmp_path: Path, kwargs, message) -> None:
    src = _rows(tmp_path / "s.jsonl", 3, 5)
    with pytest.raises(ValueError, match=message):
        mix_files([MixSource(src, 1)], tmp_path / "o.jsonl", **kwargs)


def test_mix_config_and_cli_by(tmp_path: Path, capsys) -> None:
    from convmerge.cli import main

    short = _rows(tmp_path / "short.jsonl", 100, 100)
    long = _rows(tmp_path / "long.jsonl", 100, 400)
    cfg = tmp_path / "mix.json"
    cfg.write_text(json.dumps({"sources": [{"path": str(short), "weight": 1},
                                           {"path": str(long), "weight": 1}],
                               "total": 50, "by": "chars"}))  # fmt: skip
    assert load_mix_config(cfg)[1]["by"] == "chars"
    main(["mix", str(cfg), "-o", str(tmp_path / "o.jsonl"), "--no-recipe"])
    err = capsys.readouterr().err
    assert "chars=" in err
    with pytest.raises(SystemExit) as e:
        main(["mix", "-i", f"{short}:1", "-o", str(tmp_path / "p.jsonl"), "--by", "tokens",
              "-n", "5"])  # fmt: skip
    assert e.value.code == 1
    assert "needs a tokenizer" in capsys.readouterr().err


def test_recipe_mix_by(tmp_path: Path) -> None:
    from convmerge.recipe import RecipeError
    from convmerge.recipe.schema import parse_recipe

    base = {"version": 1, "output": "o.jsonl",
            "sources": {"a": {"path": "a.jsonl", "convert": {"from": "auto"}},
                        "b": {"path": "b.jsonl", "convert": {"from": "auto"}}}}  # fmt: skip
    recipe = parse_recipe({**base, "mix": {"total": 10, "by": "chars"}}, path=tmp_path / "r.yaml")
    assert recipe.mix.by == "chars"
    with pytest.raises(RecipeError, match="needs mix.total"):
        parse_recipe({**base, "mix": {"by": "chars"}}, path=tmp_path / "r.yaml")
    with pytest.raises(RecipeError, match="needs mix.tokenizer"):
        parse_recipe({**base, "mix": {"total": 10, "by": "tokens"}}, path=tmp_path / "r.yaml")


def test_clipped_source_off_its_weight_is_warned(tmp_path: Path, caplog) -> None:
    from convmerge.mix import written_shares

    big = _rows(tmp_path / "big.jsonl", 200, 50)
    small = _rows(tmp_path / "small.jsonl", 5, 50)
    sources = [MixSource(big, 0.5), MixSource(small, 0.5)]
    with caplog.at_level("WARNING"):
        result = mix_files(sources, tmp_path / "o.jsonl", total=100)
    assert "the mix is off its weights: big.jsonl 91% (weight 50%)" in caplog.text
    assert "small.jsonl (has 5 rows, needs 50)" in caplog.text
    assert [round(x, 2) for x in written_shares(result)] == [0.91, 0.09]
    caplog.clear()
    with caplog.at_level("WARNING"):
        mix_files(sources, tmp_path / "o.jsonl", total=100, oversample=True)
        mix_files(sources, tmp_path / "o.jsonl", total=10)  # enough rows everywhere
    assert "off its weights" not in caplog.text


def test_clipped_warning_tells_same_named_sources_apart(tmp_path: Path, caplog) -> None:
    (tmp_path / "big").mkdir()
    (tmp_path / "small").mkdir()
    big = _rows(tmp_path / "big" / "converted.jsonl", 200, 50)
    small = _rows(tmp_path / "small" / "converted.jsonl", 5, 50)
    with caplog.at_level("WARNING"):
        mix_files([MixSource(big, 1), MixSource(small, 1)], tmp_path / "o.jsonl", total=100)
    assert "big/converted.jsonl 91% (weight 50%)" in caplog.text
    assert "small/converted.jsonl (has 5 rows, needs 50)" in caplog.text
