"""Tests for convmerge.mix and the ``mix`` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convmerge.mix import (
    MixSource,
    _allocate,
    load_mix_config,
    mix_files,
    write_mix_recipe,
)

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
    sources = [
        MixSource(Path("a"), 1 / 3),
        MixSource(Path("b"), 1 / 3),
        MixSource(Path("c"), 1 / 3),
    ]
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
    result = mix_files([MixSource(src_empty, 0.5), MixSource(src_ok, 0.5)], out, total=40, seed=42)

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

    main(
        [
            "mix",
            "--input",
            f"{src_a}:0.6",
            f"{src_b}:0.4",
            "--output",
            str(out),
            "--total",
            "100",
            "--seed",
            "42",
        ]
    )

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

    main(
        [
            "mix",
            "--input",
            f"{src}:1.0",
            "--output",
            str(out),
            "--no-recipe",
        ]
    )

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


@pytest.mark.parametrize(("cli_seed", "expected"), [(None, 7), ("42", 42), ("3", 3)])
def test_cli_seed_precedence_over_config(tmp_path, cli_seed, expected):
    from convmerge.cli import main

    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(20)])
    out = tmp_path / "out.jsonl"
    config_path = tmp_path / "mix.json"
    config_path.write_text(
        json.dumps({"seed": 7, "output": str(out), "sources": [{"path": str(src), "weight": 1}]})
    )
    argv = ["mix", str(config_path)]
    if cli_seed is not None:
        argv += ["--seed", cli_seed]
    main(argv)
    recipe = json.loads(out.with_suffix(".mix.json").read_text())
    assert recipe["seed"] == expected


def test_cli_seed_defaults_to_42_without_config(tmp_path):
    from convmerge.cli import main

    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(5)])
    out = tmp_path / "out.jsonl"
    main(["mix", "--input", f"{src}:1", "--output", str(out)])
    assert json.loads(out.with_suffix(".mix.json").read_text())["seed"] == 42


# --- sampler v2 (streaming) ---------------------------------------------------


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_v2_is_deterministic_and_bucketed(tmp_path, monkeypatch):
    import convmerge.mix as mixmod

    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(1000)])
    out_mem, out_bkt = tmp_path / "mem.jsonl", tmp_path / "bkt.jsonl"
    mix_files([MixSource(src, 1.0)], out_mem, total=600, seed=5)
    again = tmp_path / "again.jsonl"
    mix_files([MixSource(src, 1.0)], again, total=600, seed=5)
    assert _lines(out_mem) == _lines(again)

    # Force the on-disk bucket path: same selection, bounded buckets.
    monkeypatch.setattr(mixmod, "_BUCKET_LINES", 50)
    result = mix_files([MixSource(src, 1.0)], out_bkt, total=600, seed=5)
    assert result.total_written == 600
    rows = [json.loads(x)["x"] for x in _lines(out_bkt)]
    assert len(rows) == len(set(rows)) == 600
    assert sorted(rows) == sorted(json.loads(x)["x"] for x in _lines(out_mem))
    assert not any(p.name.startswith(".convmerge-mix-") for p in tmp_path.iterdir())


def test_v2_merge_all_keeps_every_record_once(tmp_path, monkeypatch):
    import convmerge.mix as mixmod

    monkeypatch.setattr(mixmod, "_BUCKET_LINES", 7)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [{"s": "a", "i": i} for i in range(40)])
    write_jsonl(b, [{"s": "b", "i": i} for i in range(25)])
    out = tmp_path / "out.jsonl"
    result = mix_files([MixSource(a, 1.0), MixSource(b, 9.0)], out, seed=1)
    rows = [json.loads(x) for x in _lines(out)]
    assert result.total_written == len(rows) == 65
    assert sorted((r["s"], r["i"]) for r in rows) == sorted(
        [("a", i) for i in range(40)] + [("b", i) for i in range(25)]
    )
    assert rows != sorted(rows, key=lambda r: (r["s"], r["i"]))  # shuffled


def test_v2_oversample_repeats_every_record(tmp_path):
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(10)])
    out = tmp_path / "out.jsonl"
    result = mix_files([MixSource(src, 1.0)], out, total=25, oversample=True, seed=3)
    counts = {}
    for x in _lines(out):
        k = json.loads(x)["x"]
        counts[k] = counts.get(k, 0) + 1
    assert result.total_written == 25
    assert sorted(counts.values()) == [2] * 5 + [3] * 5


def test_v2_large_fraction_uses_skip_set(tmp_path):
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(100)])
    out = tmp_path / "out.jsonl"
    mix_files([MixSource(src, 1.0)], out, total=90, seed=2)
    rows = [json.loads(x)["x"] for x in _lines(out)]
    assert len(rows) == len(set(rows)) == 90


def test_v2_skips_invalid_lines_like_v1(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text('\ufeff{"x": 1}\nnot json\n\n{"x": 2}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    result = mix_files([MixSource(src, 1.0)], out, seed=0)
    assert result.sources[0].available == 2
    assert sorted(json.loads(x)["x"] for x in _lines(out)) == [1, 2]


def test_v1_sampler_still_available_and_recorded(tmp_path):
    from convmerge.cli import main

    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(50)])
    out1, out2 = tmp_path / "v1.jsonl", tmp_path / "v1b.jsonl"
    r1 = mix_files([MixSource(src, 1.0)], out1, total=20, seed=9, sampler="v1")
    assert r1.sampler == "v1"
    main(
        [
            "mix",
            "-i",
            f"{src}:1",
            "-o",
            str(out2),
            "--total",
            "20",
            "--seed",
            "9",
            "--sampler",
            "v1",
        ]
    )
    assert _lines(out1) == _lines(out2)
    assert json.loads(out2.with_suffix(".mix.json").read_text())["sampler"] == "v1"
    with pytest.raises(ValueError, match="sampler"):
        mix_files([MixSource(src, 1.0)], out1, sampler="v3")  # type: ignore[arg-type]


def test_config_sampler_key(tmp_path):
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": 1}])
    cfg = tmp_path / "mix.json"
    cfg.write_text(json.dumps({"sampler": "v1", "sources": [{"path": str(src), "weight": 1}]}))
    assert load_mix_config(cfg)[1]["sampler"] == "v1"
    cfg.write_text(json.dumps({"sampler": "fast", "sources": [{"path": str(src), "weight": 1}]}))
    with pytest.raises(ValueError, match="sampler"):
        load_mix_config(cfg)


def _mix_digest_sources(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps({"s": "a", "i": i}) + "\n" + ("{bad\n" if i % 50 == 7 else "")
                         for i in range(400)))  # fmt: skip
    b.write_text("".join(json.dumps({"s": "b", "i": i}) + "\n" for i in range(90)))
    return [MixSource(a, 3.0), MixSource(b, 1.0)]


# sha256 of the output lines joined by "\n", as written by convmerge 1.6.5:
# the v2 sampler's output for a seed must not change between releases.
_V2_DIGESTS = {
    "skip": ({"total": 600}, "2179a491becc26ed5b330e1336b85e05403f6cd6810f2a87e3d460e7260d38fb"),
    "extra": ({"total": 200}, "5fbd1e6e7aca459bf620163d628417a375e220bf27b0a10cc4b641843afddcea"),
    "over": ({"total": 1500, "oversample": True},
             "39f4bf98c5baa7c5fd07cf1bd5673965ac54c684e2bef647e0df393083e99940"),
}  # fmt: skip


@pytest.mark.parametrize("max_open", [None, 2, 3])
@pytest.mark.parametrize("case", sorted(_V2_DIGESTS))
def test_v2_output_is_pinned_across_open_file_limits(tmp_path, monkeypatch, case, max_open):
    import hashlib

    import convmerge.mix as mixmod

    monkeypatch.setattr(mixmod, "_BUCKET_LINES", 40)  # 5 to 38 buckets
    if max_open is not None:  # buckets share files, split in several stages
        monkeypatch.setattr(mixmod, "_MAX_OPEN_FILES", max_open)
    kwargs, digest = _V2_DIGESTS[case]
    out = tmp_path / "out.jsonl"
    mix_files(_mix_digest_sources(tmp_path), out, seed=7, **kwargs)
    lines = out.read_text().splitlines()
    assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == digest


def test_v2_many_buckets_under_a_low_open_file_limit(tmp_path, monkeypatch):
    """More buckets than open files: they share files instead of failing."""
    resource = pytest.importorskip("resource")
    import os

    import convmerge.mix as mixmod

    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("needs /proc/self/fd to count open files")
    monkeypatch.setattr(mixmod, "_BUCKET_LINES", 2)
    monkeypatch.setattr(mixmod, "_MAX_OPEN_FILES", 16)
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(1000)])  # 500 buckets
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = len(os.listdir("/proc/self/fd")) + 32
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
    try:
        result = mix_files([MixSource(src, 1.0)], tmp_path / "out.jsonl", seed=3)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    rows = sorted(json.loads(x)["x"] for x in _lines(tmp_path / "out.jsonl"))
    assert result.total_written == 1000 and rows == list(range(1000))


@pytest.mark.parametrize("stage", ["write", "split"])
def test_v2_failure_leaves_no_temp_files_or_output(tmp_path, monkeypatch, stage):
    import convmerge.mix as mixmod

    monkeypatch.setattr(mixmod, "_BUCKET_LINES", 5)
    monkeypatch.setattr(mixmod, "_MAX_OPEN_FILES", 3)  # 10 buckets in 3 files
    if stage == "write":
        target, name = mixmod._Buckets, "write"
    else:
        target, name = mixmod, "_bucket_lines"
    real, calls = getattr(target, name), []

    def full(*args):
        calls.append(1)
        if len(calls) == 7:
            raise OSError(28, "No space left on device")
        return real(*args)

    monkeypatch.setattr(target, name, full)
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(50)])
    out = tmp_path / "out.jsonl"
    out.write_text("previous\n")
    with pytest.raises(OSError, match="No space"):
        mix_files([MixSource(src, 1.0)], out, seed=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.jsonl", "src.jsonl"]
    assert out.read_text() == "previous\n"


def test_v2_buckets_free_disk_while_the_output_grows(tmp_path, monkeypatch):
    import convmerge.mix as mixmod

    monkeypatch.setattr(mixmod, "_BUCKET_LINES", 10)
    monkeypatch.setattr(mixmod, "_MAX_OPEN_FILES", 4)
    left: list[int] = []
    real = mixmod._Buckets.contents

    def contents(self):
        for lines in real(self):
            left.append(sum(p.stat().st_size for p in tmp_path.glob(".convmerge-mix-*/*")))
            yield lines

    monkeypatch.setattr(mixmod._Buckets, "contents", contents)
    src = tmp_path / "src.jsonl"
    write_jsonl(src, [{"x": i} for i in range(200)])
    mix_files([MixSource(src, 1.0)], tmp_path / "out.jsonl", seed=2)
    size = (tmp_path / "out.jsonl").stat().st_size
    # Read buckets are deleted at once: nothing is left by the last bucket, and
    # splitting a shared file never needs a second copy of the whole output.
    assert len(left) == 20 and left[-1] == 0 and max(left) < 1.6 * size
