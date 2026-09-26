"""Bind adapter callables with optional chat tuning."""

from __future__ import annotations

from functools import partial

from convmerge.adapters import get_adapter
from convmerge.adapters.chat import iter_from_chat_line
from convmerge.adapters.sharegpt import iter_from_sharegpt_line
from convmerge.config import AdapterOptions


def resolve_adapter(name: str, opts: AdapterOptions | None):
    """
    Return an adapter callable ``dict -> Iterator[TrainingExample]``.

    For ``chat`` and ``auto``, applies :class:`convmerge.config.ChatAdapterOptions` when set;
    for ``sharegpt``, applies :class:`convmerge.config.SharegptAdapterOptions`.
    """
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
