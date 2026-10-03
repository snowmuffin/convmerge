"""Repeating a multi-value flag adds to the list instead of replacing it (1.5.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.cli import _build_parser, main


@pytest.mark.parametrize(
    "argv, dest, expected",
    [
        (["mix", "-i", "a:1", "-i", "b:1", "c:2"], "input", ["a:1", "b:1", "c:2"]),
        (["dedupe", "-i", "x", "-o", "y", "--keys", "k1", "--keys", "k2"], "keys", ["k1", "k2"]),
        (
            ["split", "-i", "x", "-o", "y", "--val", "0.1", "--keys", "k1", "--keys", "k2"],
            "keys",
            ["k1", "k2"],
        ),
        (["fetch", "m.yaml", "--only", "a", "--only", "b"], "only", ["a", "b"]),
        (["fetch", "gh://o/r", "--ext", "json", "--ext", "jsonl"], "ext", ["json", "jsonl"]),
        (
            ["run", "r.yaml", "--force", "a.convert", "--force", "mix"],
            "force",
            ["a.convert", "mix"],
        ),
        (["run", "r.yaml", "--force"], "force", []),
    ],
)
def test_repeated_flags_accumulate(argv, dest, expected) -> None:
    assert getattr(_build_parser().parse_args(argv), dest) == expected


@pytest.mark.parametrize(
    "argv, dest",
    [(["mix", "-o", "y"], "input"), (["dedupe", "-i", "x", "-o", "y"], "keys")],
)
def test_unset_multi_value_flags_stay_none(argv, dest) -> None:
    assert getattr(_build_parser().parse_args(argv), dest) is None


def test_mix_with_repeated_inputs_uses_every_source(tmp_path: Path) -> None:
    for name in ("a", "b"):
        turns = [{"role": "user", "content": "{}"}, {"role": "assistant", "content": "x"}]
        rows = [{"messages": [{**turns[0], "content": f"{name}{i}"}, turns[1]]} for i in range(20)]
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "o.jsonl"
    main(["mix", "-i", f"{tmp_path / 'a.jsonl'}:1", "-i", f"{tmp_path / 'b.jsonl'}:1",
          "-o", str(out), "-n", "10", "--no-recipe"])  # fmt: skip
    firsts = {json.loads(line)["messages"][0]["content"][0] for line in out.open()}
    assert firsts == {"a", "b"}


def _options(parser):
    """Every optional argument of every subcommand, as (command, action)."""
    import argparse

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield from ((f"{name}", a) for a in sub._actions if a.option_strings)


def test_every_multi_value_option_accumulates_when_repeated() -> None:
    """A flag that takes several values must keep them all when it is given
    twice; plain nargs='+' keeps only the last occurrence (the 1.5.1 bug)."""
    import argparse

    multi = [(cmd, a) for cmd, a in _options(_build_parser()) if a.nargs in ("+", "*")]
    assert multi, "no multi-value options found"
    wrong = [f"{cmd} {a.option_strings[0]}" for cmd, a in multi
             if not isinstance(a, (argparse._ExtendAction, argparse._AppendAction))]  # fmt: skip
    assert not wrong, f"repeat these flags and earlier values are lost: {wrong}"
