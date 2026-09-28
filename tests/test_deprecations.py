"""Names deprecated in 0.9 are gone in 1.0 (docs/migration-1.0.md)."""

from __future__ import annotations

import importlib
import warnings

import pytest

from convmerge._deprecation import deprecated_names, warn_deprecated

REMOVED = [
    ("convmerge.normalize", "count_turns"),
    ("convmerge.normalize", "is_single_turn"),
    ("convmerge.normalize", "detect_jsonl_shape"),
    ("convmerge.normalize", "iter_json_records"),
    ("convmerge.normalize", "load_jsonl"),
    ("convmerge.normalize", "is_uniform_schema"),
    ("convmerge.normalize", "key_frequency"),
    ("convmerge.normalize", "multi_turn_to_single_turn_record"),
    ("convmerge.normalize", "single_turn_to_multi_turn_record"),
    ("convmerge.fetch", "classify_entry"),
    ("convmerge.fetch", "sanitize_name"),
    ("convmerge.fetch", "redact_url"),
    ("convmerge.fetch", "resolve_token"),
    ("convmerge.cli", "FETCH_FILE_EXTENSIONS"),
    ("convmerge.cli", "SIDECAR_SUFFIXES"),
    ("convmerge.convert", "iter_converted_lines"),
    ("convmerge.normalize.jsonl", "load_jsonl"),
]


@pytest.mark.parametrize(("module", "name"), REMOVED)
def test_name_deprecated_in_0_9_is_removed(module: str, name: str) -> None:
    with pytest.raises(AttributeError):
        getattr(importlib.import_module(module), name)


@pytest.mark.parametrize(
    "module", ["convmerge", "convmerge.normalize", "convmerge.fetch", "convmerge.cli"]
)
def test_public_names_do_not_warn(module: str) -> None:
    mod = importlib.import_module(module)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in mod.__all__:
            getattr(mod, name)


def test_deprecation_helpers_for_1_x() -> None:
    """The helpers stay for deprecations made during 1.x (removed in 2.0)."""
    with pytest.warns(DeprecationWarning, match=r"old is deprecated.*2\.0; use new"):
        warn_deprecated("old", instead="use new", stacklevel=2)
    getattr_ = deprecated_names("pkg", {"old": ("convmerge:__version__", "use new")})
    with pytest.warns(DeprecationWarning, match=r"pkg\.old"):
        assert getattr_("old")
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        getattr_("nope")
