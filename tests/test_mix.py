"""Tests for convmerge.mix and the ``mix`` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.mix import MixSource, _allocate, load_mix_config, mix_files, write_mix_recipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# _allocate
# ---------------------------------------------------------------------------


def test_allocate_exact():
    sources = [MixSource(Path("a"), 0.6), MixSource(Path("b"), 0.4)]
    counts = _allocate(sources, 100)
    assert counts == [60, 40]
    assert sum(counts) == 100


def test_allocate_rounding_corrected():
    sources = [MixSource(Path("a"), 1 / 3), MixSource(Path("b"), 1 / 3), MixSource(Path("c"), 1 / 3)]
    counts = _allocate(sources, 100)
    assert sum(counts) == 100


def test_allocate_single_source():
    sources = [MixSource(Path("a"), 1.0)]
    assert _allocate(sources, 77) == [77]


# ---------------------------------------------------------------------------
# mix_files — basic behaviour
# ---------------------------------------------------------------------------


def test_basic_mix(tmp_path):
    src_a = tmp_path / "a.jsonl"
    src_b = tmp_path / "b.jsonl"
    write_jsonl(src_a, [{"src": "a", "i": i} for i in range(100)])
    write_jsonl(src_b, [{"src": "b", "i": i} for i in range(100)])

    out = tmp_path / "mixed.jsonl"
    result = mix_files([MixSource(src_a, 0.6), MixSource(src_b, 0.4)], out, total=100, seed=42)

    assert result.total_written == 100
    lines = read_jsonl(out)
    assert len(lines) == 100
    srcs = [r["src"] for r in lines]
    assert srcs.count("a") == 60
    assert srcs.count("b") == 40


def test_no_total_merges_all(tmp_path):
    src_a = tmp_path / "a.jsonl"
    src_b = tmp_path / "b.jsonl"
    write_jsonl(src_a, [{"x": i} for i in range(30)])
    write_jsonl(src_b, [{"x": i} for i in range(20)])

    out = tmp_path / "out.jsonl"
    result = mix_files([MixSource(src_a, 0.5), MixSource(src_b, 0.5)], out, seed=42)

    assert result.total_written == 50
    assert len(read_jsonl(out)) == 50


def test_weight_normalization(tmp_path):
    """Weights 2:1 should behave identically to 0.667:0.333."""
    src_a = tmp_path / "a.jsonl"
    src_b = tmp_path / "b.jsonl"
    write_jsonl(src_a, [{"x": i} for i in range(1000)])
    write_jsonl(src_b, [{"x": i} for i in range(1000)])

    out = tmp_path / "out.jsonl"
    result = mix_files([MixSource(src_a, 2.0), MixSource(src_b, 1.0)], out, total=99, seed=42)

    assert result.total_written == 99
    assert result.sources[0].written == 66
    assert result.sources[1].written == 33


# ---------------------------------------------------------------------------
# mix_files — reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_same_output(tmp_path):
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(1000)])

    out1 = tmp_path / "out1.jsonl"
    out2 = tmp_path / "out2.jsonl"
    mix_files([MixSource(src, 1.0)], out1, total=200, seed=7)
    mix_files([MixSource(src, 1.0)], out2, total=200, seed=7)

    assert out1.read_text() == out2.read_text()


def test_different_seeds_different_output(tmp_path):
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(1000)])

    out1 = tmp_path / "out1.jsonl"
    out2 = tmp_path / "out2.jsonl"
    mix_files([MixSource(src, 1.0)], out1, total=200, seed=1)
    mix_files([MixSource(src, 1.0)], out2, total=200, seed=2)

    assert out1.read_text() != out2.read_text()


# ---------------------------------------------------------------------------
# mix_files — clip and oversample
# ---------------------------------------------------------------------------


def test_clip_when_source_too_small(tmp_path):
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(10)])

    out = tmp_path / "out.jsonl"
    result = mix_files([MixSource(src, 1.0)], out, total=100, seed=42)

    assert result.total_written == 10
    assert result.sources[0].written == 10
    assert result.sources[0].requested == 100
    assert result.sources[0].available == 10


def test_oversample(tmp_path):
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(10)])

    out = tmp_path / "out.jsonl"
    result = mix_files([MixSource(src, 1.0)], out, total=50, seed=42, oversample=True)

    assert result.total_written == 50
    assert len(read_jsonl(out)) == 50


def test_empty_source_skipped(tmp_path):
    """Empty source contributes 0; its budget is not redistributed to others."""
    src_empty = tmp_path / "empty.jsonl"
    src_empty.write_text("")
    src_ok = tmp_path / "ok.jsonl"
    write_jsonl(src_ok, [{"x": i} for i in range(50)])

    out = tmp_path / "out.jsonl"
    result = mix_files(
        [MixSource(src_empty, 0.5), MixSource(src_ok, 0.5)], out, total=40, seed=42
    )

    assert result.sources[0].written == 0
    assert result.sources[1].written == 20  # gets its own allocation only
    assert result.total_written == 20


def test_invalid_json_lines_skipped(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text('{"x": 1}\nNOT JSON\n{"x": 2}\n')

    out = tmp_path / "out.jsonl"
    result = mix_files([MixSource(src, 1.0)], out, seed=42)

    assert result.total_written == 2


# ---------------------------------------------------------------------------
# mix_files — error handling
# ---------------------------------------------------------------------------


def test_missing_source_raises(tmp_path):
    out = tmp_path / "out.jsonl"
    with pytest.raises(FileNotFoundError):
        mix_files([MixSource(tmp_path / "ghost.jsonl", 1.0)], out, total=10, seed=42)


def test_no_sources_raises(tmp_path):
    with pytest.raises(ValueError, match="At least one source"):
        mix_files([], tmp_path / "out.jsonl", total=10, seed=42)


def test_zero_weight_raises(tmp_path):
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": 1}])
    with pytest.raises(ValueError, match="positive"):
        mix_files([MixSource(src, 0.0)], tmp_path / "out.jsonl", total=10, seed=42)


# ---------------------------------------------------------------------------
# write_mix_recipe
# ---------------------------------------------------------------------------


def test_recipe_sidecar_content(tmp_path):
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(100)])
    out = tmp_path / "out.jsonl"
    result = mix_files([MixSource(src, 1.0)], out, total=50, seed=99)

    sidecar = write_mix_recipe(result)

    assert sidecar == out.with_suffix(".mix.json")
    data = json.loads(sidecar.read_text())
    assert data["seed"] == 99
    assert data["total_written"] == 50
    assert data["version"] == 1
    assert len(data["sources"]) == 1
    assert "created_at" in data


# ---------------------------------------------------------------------------
# load_mix_config
# ---------------------------------------------------------------------------


def test_load_json_config(tmp_path):
    cfg = {
        "seed": 7,
        "total": 500,
        "output": str(tmp_path / "out.jsonl"),
        "sources": [
            {"path": str(tmp_path / "a.jsonl"), "weight": 0.7},
            {"path": str(tmp_path / "b.jsonl"), "weight": 0.3},
        ],
    }
    config_path = tmp_path / "mix.json"
    config_path.write_text(json.dumps(cfg))

    sources, options = load_mix_config(config_path)

    assert len(sources) == 2
    assert sources[0].weight == 0.7
    assert options["seed"] == 7
    assert options["total"] == 500


def test_load_config_missing_path_raises(tmp_path):
    cfg = {"sources": [{"weight": 0.5}]}
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="path"):
        load_mix_config(config_path)


def test_load_config_empty_sources_raises(tmp_path):
    cfg = {"sources": []}
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="non-empty"):
        load_mix_config(config_path)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_inline_mix(tmp_path):
    from convmerge.cli import main

    src_a = tmp_path / "a.jsonl"
    src_b = tmp_path / "b.jsonl"
    write_jsonl(src_a, [{"src": "a", "i": i} for i in range(100)])
    write_jsonl(src_b, [{"src": "b", "i": i} for i in range(100)])
    out = tmp_path / "mixed.jsonl"

    main([
        "mix",
        "--input", f"{src_a}:0.6", f"{src_b}:0.4",
        "--output", str(out),
        "--total", "100",
        "--seed", "42",
    ])

    lines = read_jsonl(out)
    assert len(lines) == 100
    assert out.with_suffix(".mix.json").is_file()


def test_cli_config_file_mix(tmp_path):
    from convmerge.cli import main

    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(200)])
    out = tmp_path / "out.jsonl"

    cfg = {
        "seed": 1,
        "total": 50,
        "output": str(out),
        "sources": [{"path": str(src), "weight": 1.0}],
    }
    config_path = tmp_path / "mix.json"
    config_path.write_text(json.dumps(cfg))

    main(["mix", str(config_path)])

    assert len(read_jsonl(out)) == 50


def test_cli_no_recipe_flag(tmp_path):
    from convmerge.cli import main

    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(50)])
    out = tmp_path / "out.jsonl"

    main([
        "mix",
        "--input", f"{src}:1.0",
        "--output", str(out),
        "--no-recipe",
    ])

    assert out.is_file()
    assert not out.with_suffix(".mix.json").exists()


def test_cli_missing_output_exits(tmp_path):
    from convmerge.cli import main

    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": 1}])

    with pytest.raises(SystemExit) as exc:
        main(["mix", "--input", f"{src}:1.0"])
    assert exc.value.code != 0


def test_cli_bad_spec_exits(tmp_path):
    from convmerge.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["mix", "--input", "no-colon-here", "--output", str(tmp_path / "out.jsonl")])
    assert exc.value.code != 0
