"""convmerge — merge heterogeneous sources into a single LLM training format.

The names in ``__all__`` are the public API (documented in ``docs/api.md``):
they keep working across minor versions, with changes announced in the
changelog first. Anything else — modules or names starting with ``_`` — may
change without notice. Names are imported lazily, so ``import convmerge``
stays cheap.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.8.0"

_EXPORTS: dict[str, str] = {
    # convert pipeline
    "convert_file": "convmerge.convert",
    "convert_with_config": "convmerge.convert",
    "validate_file": "convmerge.convert",
    "ConvertStats": "convmerge.convert",
    "InvalidExampleError": "convmerge.convert",
    "ConvertConfig": "convmerge.config",
    "AdapterOptions": "convmerge.config",
    "ChatAdapterOptions": "convmerge.config",
    "SharegptAdapterOptions": "convmerge.config",
    "build_convert_config": "convmerge.config",
    "EmitOptions": "convmerge.emitters",
    "UnrepresentableExample": "convmerge.emitters",
    # data model
    "TrainingExample": "convmerge.models",
    "ChatMessage": "convmerge.models",
    "ContentPart": "convmerge.models",
    "ToolCall": "convmerge.models",
    # validation
    "validate_example": "convmerge.validate",
    # extension points
    "register_adapter": "convmerge.adapters",
    "register_emitter": "convmerge.emitters",
    "available_adapters": "convmerge.adapters",
    "available_formats": "convmerge.emitters",
    # other commands
    "mix_files": "convmerge.mix",
    "MixSource": "convmerge.mix",
    "deduplicate_jsonl": "convmerge.normalize.dedup",
    "DedupeStats": "convmerge.normalize.dedup",
    "normalize_to_jsonl": "convmerge.normalize.jsonl",
    "profile_schema": "convmerge.normalize.schema",
    "iter_jsonl": "convmerge.io",
}

__all__ = [
    "__version__",
    "AdapterOptions",
    "ChatAdapterOptions",
    "ChatMessage",
    "ContentPart",
    "ConvertConfig",
    "ConvertStats",
    "DedupeStats",
    "EmitOptions",
    "InvalidExampleError",
    "MixSource",
    "SharegptAdapterOptions",
    "ToolCall",
    "TrainingExample",
    "UnrepresentableExample",
    "available_adapters",
    "available_formats",
    "build_convert_config",
    "convert_file",
    "convert_with_config",
    "deduplicate_jsonl",
    "iter_jsonl",
    "mix_files",
    "normalize_to_jsonl",
    "profile_schema",
    "register_adapter",
    "register_emitter",
    "validate_example",
    "validate_file",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'convmerge' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from convmerge.adapters import available_adapters, register_adapter
    from convmerge.config import (
        AdapterOptions,
        ChatAdapterOptions,
        ConvertConfig,
        SharegptAdapterOptions,
        build_convert_config,
    )
    from convmerge.convert import (
        ConvertStats,
        InvalidExampleError,
        convert_file,
        convert_with_config,
        validate_file,
    )
    from convmerge.emitters import (
        EmitOptions,
        UnrepresentableExample,
        available_formats,
        register_emitter,
    )
    from convmerge.io import iter_jsonl
    from convmerge.mix import MixSource, mix_files
    from convmerge.models import ChatMessage, ContentPart, ToolCall, TrainingExample
    from convmerge.normalize.dedup import DedupeStats, deduplicate_jsonl
    from convmerge.normalize.jsonl import normalize_to_jsonl
    from convmerge.normalize.schema import profile_schema
    from convmerge.validate import validate_example
