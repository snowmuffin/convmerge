"""CLI errors name the flags the user typed, and typos are not ignored (1.6.1)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from convmerge.cli import main
from convmerge.mix import MixSource, mix_files

ROW = json.dumps({"messages": [{"role": "user", "content": "q"},
                               {"role": "assistant", "content": "a"}]})  # fmt: skip


def _src(path: Path, n: int = 3) -> Path:
    path.write_text((ROW + "\n") * n, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--max-tokens", "5", "-n", "4"], "--max-tokens needs --by tokens"),
        (["--by-sample", "2", "-n", "4"], "--by-sample needs --by chars or --by tokens"),
        (["--by", "chars"], "--by chars needs a total (-n/--total N)"),
        (["--by", "tokens", "-n", "4"], "--by tokens needs --tokenizer NAME"),
        (["--by", "chars", "-n", "4", "--sampler", "v1"], "--by chars needs the v2 sampler"),
        (["--tokenizer", "gpt2", "-n", "4"], "--tokenizer needs --by tokens"),
    ],
)
def test_mix_errors_name_the_flags(extra, message, tmp_path, capsys) -> None:
    a = _src(tmp_path / "a.jsonl")
    with pytest.raises(SystemExit) as exc:
        main(["mix", "-i", f"{a}:1", "-o", str(tmp_path / "o.jsonl"), "--no-recipe", *extra])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    assert "by='" not in err and "max_tokens" not in err and "by_sample" not in err


def test_mix_errors_name_config_keys(tmp_path, capsys) -> None:
    a = _src(tmp_path / "a.jsonl")
    config = tmp_path / "mix.json"
    config.write_text(json.dumps({"sources": [{"path": str(a), "weight": 1}],
                                  "output": str(tmp_path / "o.jsonl"),
                                  "total": 4, "max_tokens": 5}), encoding="utf-8")  # fmt: skip
    with pytest.raises(SystemExit) as exc:
        main(["mix", str(config), "--no-recipe"])
    assert exc.value.code == 2
    assert "'max_tokens' in the config needs --by tokens" in capsys.readouterr().err


def test_clip_warning_names_cli_and_recipe_options(tmp_path, caplog) -> None:
    a, b = _src(tmp_path / "a.jsonl", 10), _src(tmp_path / "b.jsonl", 1)
    with caplog.at_level(logging.WARNING, logger="convmerge"):
        mix_files([MixSource(a, 1), MixSource(b, 1)], tmp_path / "o.jsonl", total=10)
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "--oversample, or mix.oversample in a recipe" in message
    assert "-n/--total, or mix.total" in message


def _recipe(tmp_path: Path) -> Path:
    _src(tmp_path / "a.jsonl")
    recipe = tmp_path / "r.json"
    recipe.write_text(json.dumps({
        "version": 1, "output": "out.jsonl",
        "sources": {"a": {"path": "a.jsonl", "normalize": False, "convert": {"from": "auto"}}},
    }), encoding="utf-8")  # fmt: skip
    return recipe


def test_run_force_rejects_unknown_names(tmp_path, capsys) -> None:
    recipe = _recipe(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["run", str(recipe), "--plan", "--force", "convrt"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "'convrt' (did you mean 'convert'?)" in err and "a.convert" in err


@pytest.mark.parametrize("force", [[], ["a"], ["convert"], ["a.convert"]])
def test_run_force_accepts_steps_sources_and_kinds(force, tmp_path, capsys) -> None:
    recipe = _recipe(tmp_path)
    main(["run", str(recipe), "--plan", "--force", *force])
    assert "forced" in capsys.readouterr().out
