"""Bind adapter callables with optional chat tuning."""

from __future__ import annotations

from functools import partial

from convmerge.adapters import get_adapter
from convmerge.adapters.chat import iter_from_chat_line
from convmerge.adapters.preference import apply_preference, is_preference_record, iter_pairs
from convmerge.adapters.sharegpt import iter_from_sharegpt_line
from convmerge.config import AdapterOptions
from convmerge.models import TrainingExample
from convmerge.validate import validate_example


def resolve_adapter(name: str, opts: AdapterOptions | None, *, pairs: bool = False):
    """
    Return an adapter callable ``dict -> Iterator[TrainingExample]``.

    For ``chat`` and ``auto``, applies :class:`convmerge.config.ChatAdapterOptions` when set;
    for ``sharegpt``, applies :class:`convmerge.config.SharegptAdapterOptions`. With
    ``opts.preference``, preference records are folded into plain SFT records first.
    ``pairs=True`` (the ``preference`` output format) keeps both answers instead.
    Without either, a preference record the adapter cannot turn into a valid
    example is tagged ``preference_record`` so the drop report says what to do.
    """
    adapter = _resolve(name, opts)
    preference = opts.preference if opts is not None else None
    if pairs:
        if preference:
            raise ValueError(
                f"preference={preference!r} folds pairs into SFT examples; "
                "leave it out to write chosen/rejected pairs"
            )
        return partial(iter_pairs, adapter=adapter)
    if preference:
        return partial(_with_preference, adapter=adapter, which=preference)
    return partial(_flag_preference, adapter=adapter)


def _with_preference(record, *, adapter, which):
    return adapter(apply_preference(record, which))


def _flag_preference(record, *, adapter):
    if "rejected" not in record or not is_preference_record(record):
        return adapter(record)
    return _flagged(record, adapter)


def _flagged(record, adapter):
    found = False
    for example in adapter(record):
        found = True
        if validate_example(example):
            example.issues = [*example.issues, "preference_record"]
        yield example
    if not found:
        yield TrainingExample(issues=["preference_record"])


def _resolve(name: str, opts: AdapterOptions | None):
    if name == "sharegpt" and opts is not None and opts.sharegpt is not None:
        get_adapter(name)  # validates the name like the other branches
        return partial(iter_from_sharegpt_line, turn_mode=opts.sharegpt.turn_mode)
    if opts is None or opts.chat is None or name not in ("chat", "auto"):
        return get_adapter(name)
    o = opts.chat
    return partial(
        iter_from_chat_line,
        conversation_keys=o.conversation_keys,
        role_keys=o.role_keys,
        content_keys=o.content_keys,
        role_map=o.role_map,
        pairwise_mode=o.pairwise_mode,
        instruction_keys=o.instruction_keys,
        output_keys=o.output_keys,
        input_keys=o.input_keys,
    )
