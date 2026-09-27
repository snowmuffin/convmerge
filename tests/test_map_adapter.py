"""--from map: dotted-path field mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge import ConvertStats, MapSpec, build_convert_config, convert_with_config
from convmerge.adapters.mapped import Path as MapPath
from convmerge.adapters.mapped import iter_from_mapped_line
from convmerge.cli import main
from convmerge.models import ChatMessage
from convmerge.preset import load_convert_preset
from convmerge.recipe import RecipeError, parse_recipe

AIHUB = {
    "info": {"topic": "금융 상담"},
    "dialogue": [
        {"utterances": [{"speaker": "고객", "text": "금리가 궁금해요."},
                        {"speaker": "상담사", "text": "연 3.5%입니다."}]},
        {"utterances": [{"speaker": "고객", "text": "해지하면요?"},
                        {"speaker": "상담사", "text": "50%가 적용됩니다."}]},
    ],
}  # fmt: skip
TURNS = {
    "turns": "dialogue[].utterances[]",
    "role": "speaker",
    "content": "text",
    "role_map": {"고객": "user", "상담사": "assistant"},
    "system": "info.topic",
}


def _one(record: dict, mapping: dict, **kw):
    (ex,) = list(iter_from_mapped_line(record, spec=MapSpec.from_mapping(mapping), **kw))
    return ex


def test_paths() -> None:
    doc = {"a": {"b": [{"c": 1}, {"c": 2}], "x.y": "dotted"}, "l": [[1, 2], [3]]}
    assert MapPath.parse("a.b[1].c").get(doc) == 2
    assert MapPath.parse("a.b[-1].c").get(doc) == 2
    assert MapPath.parse("a.b[].c").values(doc) == [1, 2]
    assert MapPath.parse('a."x.y"').get(doc) == "dotted"
    assert MapPath.parse("l[][]").values(doc) == [1, 2, 3]
    assert MapPath.parse("a.missing").values(doc) == []
    assert MapPath.parse("a.b[9]").values(doc) == []
    for bad in ("", "a..b", "a[x]", 'a."b'):
        with pytest.raises(ValueError, match="map"):
            MapPath.parse(bad)


def test_turns_mode() -> None:
    ex = _one(AIHUB, TURNS)
    assert ex.messages == [
        ChatMessage("system", "금융 상담"),
        ChatMessage("user", "금리가 궁금해요."),
        ChatMessage("assistant", "연 3.5%입니다."),
        ChatMessage("user", "해지하면요?"),
        ChatMessage("assistant", "50%가 적용됩니다."),
    ]
    assert ex.meta == {"source": "map"}
    # A plain list of turns, default role / content keys and role names.
    ex = _one({"log": [{"role": "Human", "content": "hi"}, {"role": "GPT", "content": "yo"}]},
              {"turns": "log"})  # fmt: skip
    assert [m.role for m in ex.messages] == ["user", "assistant"]


def test_flat_mode_with_reasoning_and_tools() -> None:
    record = {
        "id": 5,
        "q": {"text": "2+2?"},
        "a": {"text": "4", "why": "Two plus two."},
        "fns": [{"name": "calc", "parameters": {}}],
    }
    ex = _one(record, {"user": "q.text", "assistant": "a.text", "reasoning": "a.why",
                       "tools": "fns"})  # fmt: skip
    assert ex.messages == [
        ChatMessage("user", "2+2?"),
        ChatMessage("assistant", "4", reasoning="Two plus two."),
    ]
    assert ex.meta == {"source": "map", "id": 5}
    assert ex.tools == [{"type": "function", "function": {"name": "calc", "parameters": {}}}]
    assert _one({"q": 3, "a": 4}, {"user": "q", "assistant": "a"}).messages[0].content == "3"


def test_preference_pairs() -> None:
    record = {"q": "sky?", "good": "blue", "bad": "green", "r": "light scatters"}
    mapping = {"user": "q", "chosen": "good", "rejected": "bad", "reasoning": "r"}
    ex = _one(record, mapping)
    assert ex.messages[-1] == ChatMessage("assistant", "blue", reasoning="light scatters")
    assert ex.rejected == [ChatMessage("user", "sky?"), ChatMessage("assistant", "green")]
    assert _one(record, mapping, preference="rejected").messages[-1].content == "green"
    assert _one(record, mapping, preference="chosen").rejected is None


def test_missing_paths_are_reported() -> None:
    ex = _one({"q": "x"}, {"user": "q", "assistant": "a.text"})
    assert ex.messages == [] and ex.issues == ["map_path_missing"]
    assert ex.meta["missing"] == "a.text"
    assert _one({"other": []}, TURNS).issues == ["map_path_missing"]


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"user": "q"}, "give 'turns'"),
        ({"turns": "t", "user": "q"}, "cannot be combined"),
        ({"user": "q", "chosen": "c"}, "go together"),
        ({"user": "q", "assistant": "a", "role": "r"}, "only applies with 'turns'"),
        ({"user": "q", "assistant": "a", "speaker": "s"}, "unknown key"),
        ({"turns": "t", "role_map": ["x"]}, "role_map"),
        ([], "expected a mapping"),
    ],
)
def test_bad_specs(mapping, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MapSpec.from_mapping(mapping)


def _src(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "in.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return p


def test_convert_cli_preset_recipe_and_workers(tmp_path: Path, capsys) -> None:
    src = _src(tmp_path, [AIHUB, {"nothing": 1}] * 3)
    out = tmp_path / "o.jsonl"
    main(["convert", "-i", str(src), "-o", str(out), "--from", "map", "--format", "sharegpt",
          "--adapter-kwargs", json.dumps({"map": TURNS})])  # fmt: skip
    err = capsys.readouterr().err
    assert "wrote 3 examples" in err and "map_path_missing" in err
    assert json.loads(out.read_text().splitlines()[0])["system"] == "금융 상담"

    cfg = build_convert_config(adapter="map", output_format="messages",
                               adapter_kwargs_json=json.dumps({"map": TURNS}))  # fmt: skip
    single, parallel = ConvertStats(), ConvertStats()
    convert_with_config(src, tmp_path / "a.jsonl", cfg, stats=single)
    convert_with_config(src, tmp_path / "b.jsonl", cfg, stats=parallel, workers=2)
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert single.drop_reasons == parallel.drop_reasons == {"map_path_missing": 3}

    preset = tmp_path / "p.json"
    preset.write_text(json.dumps({"adapter": "map", "output_format": "messages",
                                  "adapter_options": {"map": TURNS}}))  # fmt: skip
    assert load_convert_preset(preset).adapter_options.map == MapSpec.from_mapping(TURNS)

    recipe = parse_recipe(
        {"output": "o.jsonl", "sources": {"a": {"path": "in.jsonl", "convert": {"map": TURNS}}}},
        path=tmp_path / "r.yaml",
    )
    conv = recipe.sources["a"].convert
    assert conv.adapter == "map" and conv.adapter_kwargs == {"map": TURNS}
    with pytest.raises(RecipeError, match="map"):
        parse_recipe(
            {"output": "o.jsonl",
             "sources": {"a": {"path": "in.jsonl", "convert": {"map": {"user": "q"}}}}},
            path=tmp_path / "r.yaml",
        )  # fmt: skip


def test_map_without_spec_is_a_config_error(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as e:
        main(["convert", "-i", str(_src(tmp_path, [AIHUB])), "-o", str(tmp_path / "o.jsonl"),
              "--from", "map", "--format", "messages"])  # fmt: skip
    assert e.value.code == 2
    assert "needs a field mapping" in capsys.readouterr().err
