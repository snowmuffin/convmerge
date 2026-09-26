"""Deprecated names keep working in 0.x and warn (docs/stability.md)."""

from __future__ import annotations

import importlib
import warnings

import pytest

DEPRECATED = [
    ("convmerge.normalize", "count_turns", "convmerge.normalize.turns"),
    ("convmerge.normalize", "is_single_turn", "convmerge.normalize.turns"),
    ("convmerge.normalize", "detect_jsonl_shape", "convmerge.normalize.jsonl"),
    ("convmerge.normalize", "iter_json_records", "convmerge.normalize.jsonl"),
    ("convmerge.normalize", "load_jsonl", "convmerge.normalize.jsonl"),
    ("convmerge.normalize", "is_uniform_schema", "convmerge.normalize.schema"),
    ("convmerge.normalize", "key_frequency", "convmerge.normalize.schema"),
    (
        "convmerge.normalize",
        "multi_turn_to_single_turn_record",
        "convmerge.normalize.convert_turns",
    ),
    (
        "convmerge.normalize",
        "single_turn_to_multi_turn_record",
        "convmerge.normalize.convert_turns",
    ),
    ("convmerge.fetch", "classify_entry", "convmerge.fetch.manifest"),
    ("convmerge.fetch", "sanitize_name", "convmerge.fetch.manifest"),
    ("convmerge.fetch", "redact_url", "convmerge.fetch.auth"),
    ("convmerge.fetch", "resolve_token", "convmerge.fetch.auth"),
]


@pytest.mark.parametrize(("module", "name", "home"), DEPRECATED)
def test_deprecated_name_warns_and_still_works(module: str, name: str, home: str) -> None:
    mod = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=rf"{module}\.{name} is deprecated.*1\.0"):
        value = getattr(mod, name)
    assert value is getattr(importlib.import_module(home), name)
    assert name not in mod.__all__


def test_deprecated_cli_constants() -> None:
    import convmerge.cli as cli
    from convmerge.normalize.files import NORMALIZE_EXTENSIONS, SIDECAR_SUFFIXES

    with pytest.warns(DeprecationWarning, match="exports only main"):
        assert cli.FETCH_FILE_EXTENSIONS is NORMALIZE_EXTENSIONS
    with pytest.warns(DeprecationWarning):
        assert cli.SIDECAR_SUFFIXES is SIDECAR_SUFFIXES
    assert cli.__all__ == ["main"]


def test_deprecated_iter_converted_lines() -> None:
    import convmerge.convert as conv

    with pytest.warns(DeprecationWarning, match="iter_converted_lines"):
        fn = conv.iter_converted_lines
    assert fn is conv._iter_converted_lines


def test_warning_points_at_the_caller() -> None:
    with pytest.warns(DeprecationWarning) as record:
        from convmerge.normalize import count_turns  # noqa: F401
    assert record[0].filename == __file__


@pytest.mark.parametrize("module", ["convmerge", "convmerge.normalize", "convmerge.fetch"])
def test_public_names_do_not_warn(module: str) -> None:
    mod = importlib.import_module(module)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in mod.__all__:
            getattr(mod, name)


def test_unknown_name_is_still_an_attribute_error() -> None:
    import convmerge.normalize

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        convmerge.normalize.nope  # noqa: B018
