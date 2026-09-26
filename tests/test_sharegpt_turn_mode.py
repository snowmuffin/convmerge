"""sharegpt adapter turn_mode (pairs | full) and its deprecation path."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from convmerge.adapters.sharegpt import iter_from_sharegpt_line
from convmerge.cli import main
from convmerge.config import AdapterOptions, SharegptAdapterOptions, build_convert_config
from convmerge.convert import convert_file
from convmerge.preset import load_convert_preset, validate_preset_file

MULTI = {
    "conversations": [
        {"from": "system", "value": "SYS"},
        {"from": "human", "value": "q1"},
        {"from": "gpt", "value": "a1"},
        {"from": "human", "value": "q2"},
        {"from": "gpt", "value": " a2 "},
    ]
}
SINGLE = {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}


def _roles_contents(examples):
    return [[(m.role, m.content) for m in ex.messages] for ex in examples]


def test_full_keeps_whole_conversation() -> None:
    out = list(iter_from_sharegpt_line(MULTI, turn_mode="full"))
    assert _roles_contents(out) == [
        [
            ("system", "SYS"),
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
            ("assistant", "a2"),
        ]
    ]


def test_full_drops_empty_turns() -> None:
    rec = {
        "conversations": [
            {"from": "system", "value": ""},
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
        ]
    }
    assert _roles_contents(iter_from_sharegpt_line(rec, turn_mode="full")) == [
        [("user", "q"), ("assistant", "a")]
    ]
    # Incomplete conversations are still emitted; validation decides what to drop.
    only_user = {"conversations": [{"from": "human", "value": "q"}]}
    assert _roles_contents(iter_from_sharegpt_line(only_user, turn_mode="full")) == [
        [("user", "q")]
    ]


def test_pairs_is_legacy_behavior_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = list(iter_from_sharegpt_line(MULTI, turn_mode="pairs"))
    assert _roles_contents(out) == [
        [("user", "q1"), ("assistant", "a1")],
        [("user", "q2"), ("assistant", "a2")],
    ]


def test_default_is_full_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        default = list(iter_from_sharegpt_line(MULTI))
    assert _roles_contents(default) == _roles_contents(
        iter_from_sharegpt_line(MULTI, turn_mode="full")
    )


def test_non_string_value_does_not_crash() -> None:
    rec = {"conversations": [{"from": "human", "value": 5}, {"from": "gpt", "value": "a"}]}
    assert _roles_contents(iter_from_sharegpt_line(rec, turn_mode="pairs")) == [
        [("user", ""), ("assistant", "a")]
    ]
    assert _roles_contents(iter_from_sharegpt_line(rec, turn_mode="full")) == [[("assistant", "a")]]


def test_unknown_turn_mode_raises() -> None:
    with pytest.raises(ValueError, match="turn_mode"):
        list(iter_from_sharegpt_line(SINGLE, turn_mode="nope"))


def _write(path: Path, *records: dict) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_convert_file_with_sharegpt_options(tmp_path: Path) -> None:
    src = _write(tmp_path / "in.jsonl", MULTI)
    out = tmp_path / "out.jsonl"
    opts = AdapterOptions(sharegpt=SharegptAdapterOptions(turn_mode="full"))
    _, written = convert_file(
        src, out, adapter_name="sharegpt", output_format="messages", adapter_options=opts
    )
    assert written == 1
    assert json.loads(out.read_text())["messages"][0] == {"role": "system", "content": "SYS"}


def test_build_convert_config_sharegpt_layers(tmp_path: Path) -> None:
    preset = tmp_path / "p.json"
    preset.write_text(
        json.dumps(
            {
                "adapter": "sharegpt",
                "output_format": "messages",
                "adapter_options": {"sharegpt": {"turn_mode": "pairs"}},
            }
        ),
        encoding="utf-8",
    )
    assert load_convert_preset(preset).adapter_options.sharegpt.turn_mode == "pairs"
    validate_preset_file(preset)
    cfg = build_convert_config(
        preset_path=preset, adapter_kwargs_json='{"sharegpt": {"turn_mode": "full"}}'
    )
    assert cfg.adapter_options.sharegpt.turn_mode == "full"
    assert cfg.adapter_options.chat is None


@pytest.mark.parametrize(
    "kwargs",
    ['{"sharegpt": {"turn_mode": "nope"}}', '{"sharegpt": {"mode": "full"}}', '{"sharegpt": 1}'],
)
def test_build_convert_config_rejects_bad_sharegpt_options(kwargs: str) -> None:
    with pytest.raises(ValueError, match="sharegpt"):
        build_convert_config(
            adapter="sharegpt", output_format="messages", adapter_kwargs_json=kwargs
        )


def test_cli_pairs_mode(tmp_path: Path, capsys) -> None:
    src = _write(tmp_path / "in.jsonl", MULTI)
    main(
        [
            "convert",
            "-i",
            str(src),
            "-o",
            str(tmp_path / "o.jsonl"),
            "--from",
            "sharegpt",
            "-f",
            "messages",
            "--adapter-kwargs",
            '{"sharegpt": {"turn_mode": "pairs"}}',
        ]
    )
    assert "wrote 2 examples" in capsys.readouterr().err
