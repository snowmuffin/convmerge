"""convmerge — merge heterogeneous sources into a single LLM training format.

The names in ``__all__`` are the public API (documented in ``docs/api.md``),
together with the ``convmerge.recipe`` and ``convmerge.fetch`` modules'
``__all__``. They follow the stability policy in ``docs/stability.md``.
Anything else — other modules, names starting with ``_`` — is internal and
may change without notice. Names are imported lazily, so ``import convmerge``
stays cheap.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "1.6.4"

_EXPORTS: dict[str, str] = {
    # convert pipeline
    "convert_file": "convmerge.convert",
    "convert_records": "convmerge.convert",
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
    "TransformOptions": "convmerge.transforms",
    "MapSpec": "convmerge.adapters.mapped",
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
    "MixResult": "convmerge.mix",
    "deduplicate_jsonl": "convmerge.normalize.dedup",
    "DedupeStats": "convmerge.normalize.dedup",
    "normalize_to_jsonl": "convmerge.normalize.jsonl",
    "profile_schema": "convmerge.normalize.schema",
    "split_by_turns": "convmerge.normalize.turns",
    "split_jsonl": "convmerge.split",
    "SplitStats": "convmerge.split",
    "check_tokens": "convmerge.tokens",
    "TokenStats": "convmerge.tokens",
    # quality
    "filter_jsonl": "convmerge.quality",
    "FilterSpec": "convmerge.quality",
    "FilterStats": "convmerge.quality",
    "decontaminate_jsonl": "convmerge.decontam",
    "build_index": "convmerge.decontam",
    "EvalSource": "convmerge.decontam",
    "DecontamStats": "convmerge.decontam",
    "deduplicate_near_jsonl": "convmerge.normalize.near_dedup",
    "NearDedupeStats": "convmerge.normalize.near_dedup",
    "analyze_turn_distribution": "convmerge.normalize.turns",
    "iter_jsonl": "convmerge.io",
    "JsonlLine": "convmerge.io",
    "JsonlDecodeError": "convmerge.io",
    "ReadStats": "convmerge.io",
}

__all__ = [
    "__version__",
    "AdapterOptions",
    "ChatAdapterOptions",
    "ChatMessage",
    "ContentPart",
    "ConvertConfig",
    "ConvertStats",
    "DecontamStats",
    "DedupeStats",
    "EmitOptions",
    "EvalSource",
    "FilterSpec",
    "FilterStats",
    "InvalidExampleError",
    "JsonlDecodeError",
    "JsonlLine",
    "MapSpec",
    "MixResult",
    "MixSource",
    "NearDedupeStats",
    "ReadStats",
    "SharegptAdapterOptions",
    "SplitStats",
    "TokenStats",
    "ToolCall",
    "TrainingExample",
    "TransformOptions",
    "UnrepresentableExample",
    "analyze_turn_distribution",
    "available_adapters",
    "available_formats",
    "build_convert_config",
    "build_index",
    "check_tokens",
    "convert_file",
    "convert_records",
    "convert_with_config",
    "decontaminate_jsonl",
    "deduplicate_jsonl",
    "deduplicate_near_jsonl",
    "filter_jsonl",
    "iter_jsonl",
    "mix_files",
    "normalize_to_jsonl",
    "profile_schema",
    "register_adapter",
    "register_emitter",
    "split_by_turns",
    "split_jsonl",
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
    from convmerge.adapters.mapped import MapSpec
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
        convert_records,
        convert_with_config,
        validate_file,
    )
    from convmerge.decontam import DecontamStats, EvalSource, build_index, decontaminate_jsonl
    from convmerge.emitters import (
        EmitOptions,
        UnrepresentableExample,
        available_formats,
        register_emitter,
    )
    from convmerge.io import JsonlDecodeError, JsonlLine, ReadStats, iter_jsonl
    from convmerge.mix import MixResult, MixSource, mix_files
    from convmerge.models import ChatMessage, ContentPart, ToolCall, TrainingExample
    from convmerge.normalize.dedup import DedupeStats, deduplicate_jsonl
    from convmerge.normalize.jsonl import normalize_to_jsonl
    from convmerge.normalize.near_dedup import NearDedupeStats, deduplicate_near_jsonl
    from convmerge.normalize.schema import profile_schema
    from convmerge.normalize.turns import analyze_turn_distribution, split_by_turns
    from convmerge.quality import FilterSpec, FilterStats, filter_jsonl
    from convmerge.split import SplitStats, split_jsonl
    from convmerge.tokens import TokenStats, check_tokens
    from convmerge.transforms import TransformOptions
    from convmerge.validate import validate_example
