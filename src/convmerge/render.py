"""Chat-template-rendered text for trainers that take plain text (``convmerge render``).

Every row is read the way ``convert`` and ``tokens`` read it (any supported
layout) and rendered with the target model's chat template, as the trainer
would render it, into ``{"text": ...}``. The row's ``meta`` is kept.

With ``assistant_spans``, each row also gets ``assistant_spans``: the
``[start, end)`` character offsets in ``text`` of every assistant turn the
loss should cover (its answer and the end-of-turn text the template adds),
for trainers that mask everything else. A turn marked ``"train": false`` is
left out. Offsets come from rendering the conversation up to each turn, so
they need a template whose output for a shorter conversation is a prefix of
its output for a longer one; rows where that does not hold (a template that
drops earlier reasoning, for example) are dropped and counted, never written
with a wrong mask.

Render last: ``filter``, ``decontam`` and ``tokens`` read conversations, not
rendered text. Preference pairs are not rendered (preference trainers apply
the template themselves) and are counted. Needs ``pip install
"convmerge[tokens]"`` (``transformers``; no PyTorch).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from convmerge.io import ReadStats, iter_jsonl, refuse_overwrite

__all__ = ["RenderStats", "render_jsonl"]

_REASON_CHARS = 160


@dataclass
class RenderStats:
    """Counts for one :func:`render_jsonl` call."""

    rows: int = 0
    written: int = 0
    unreadable: int = 0
    invalid_json: int = 0
    pairs: int = 0
    """Preference rows, which are not rendered."""
    template_errors: dict[str, int] = field(default_factory=dict)
    unstable_spans: int = 0
    """Rows dropped because the template did not render each turn as a prefix."""
    no_trained_turn: int = 0
    """Rows dropped with ``assistant_spans`` because no assistant turn is trained."""
    tokenizer: str | None = None

    def to_report(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer,
            "rows": self.rows,
            "written": self.written,
            "unreadable": self.unreadable,
            "invalid_json": self.invalid_json,
            "pairs": self.pairs,
            "template_errors": dict(sorted(self.template_errors.items())),
            "unstable_spans": self.unstable_spans,
            "no_trained_turn": self.no_trained_turn,
        }


def render_jsonl(
    src: str | Path,
    dst: str | Path,
    *,
    tokenizer: Any,
    chat_template: str | None = None,
    assistant_spans: bool = False,
    encoding: str = "utf-8",
    stats: RenderStats | None = None,
) -> RenderStats:
    """Render every conversation in ``src`` to ``{"text": ...}`` rows in ``dst``.

    ``tokenizer`` is a model name or path, or a loaded ``transformers``
    tokenizer; ``chat_template`` (Jinja source) overrides its template.
    Rows that cannot be rendered are dropped and counted in the stats.
    """
    from convmerge._output import atomic_text_writer
    from convmerge.adapter_resolve import resolve_adapter
    from convmerge.emitters import _message_dict
    from convmerge.tokens import (
        _reasoning_key,
        _require_template_support,
        _stored_sides,
        load_tokenizer,
    )

    refuse_overwrite([src], [dst])
    _require_template_support()
    tok = load_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
    template = chat_template if chat_template is not None else getattr(tok, "chat_template", None)
    if not template:
        raise ValueError(
            "the tokenizer has no chat template; pass one with --chat-template (a Jinja file)"
        )
    st = stats if stats is not None else RenderStats()
    st.tokenizer = tokenizer if isinstance(tokenizer, str) else getattr(tok, "name_or_path", None)
    adapter = resolve_adapter("auto", None, pairs=True)

    def render(dicts: list[dict[str, Any]], tools: Any, prompt: bool = False) -> str:
        text = tok.apply_chat_template(
            dicts, tools=tools, chat_template=template, tokenize=False,
            add_generation_prompt=prompt,
        )  # fmt: skip
        if not isinstance(text, str):
            raise TypeError("the template did not render text")
        return text

    read = ReadStats()
    with atomic_text_writer(dst) as out:
        for line in iter_jsonl(src, encoding=encoding, stats=read):
            st.rows += 1
            value = line.value
            examples = list(adapter(value)) if isinstance(value, dict) else []
            if not examples:
                st.unreadable += 1
                continue
            if any(ex.rejected is not None for ex in examples):
                st.pairs += 1
                continue
            stored = _stored_sides(value) if len(examples) == 1 else None
            key = _reasoning_key(value)
            meta = value.get("meta") if isinstance(value, dict) else None
            for ex in examples:
                if stored is not None:
                    dicts, tools = stored[0], value.get("tools", ex.tools)
                else:
                    dicts = []
                    for m in ex.messages:
                        d = _message_dict(m, "object", reasoning_key=key, tool_content="null")
                        if m.train is not None:
                            d["train"] = m.train
                        dicts.append(d)
                    tools = ex.tools
                try:
                    text = render(dicts, tools)
                    spans = _spans(dicts, tools, text, render) if assistant_spans else None
                except ImportError:
                    raise  # a missing dependency, not a problem with this row
                except Exception as e:  # noqa: BLE001 - templates raise anything
                    error = f"{type(e).__name__}: {e}"[:_REASON_CHARS]
                    st.template_errors[error] = st.template_errors.get(error, 0) + 1
                    continue
                row: dict[str, Any] = {"text": text}
                if assistant_spans:
                    if spans is None:
                        st.unstable_spans += 1
                        continue
                    if not spans:
                        st.no_trained_turn += 1
                        continue
                    row["assistant_spans"] = spans
                if isinstance(meta, dict):
                    row["meta"] = meta
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                st.written += 1
    st.invalid_json = read.invalid_json
    return st


def _spans(
    dicts: list[dict[str, Any]], tools: Any, text: str, render: Any
) -> list[list[int]] | None:
    """``[start, end)`` of each trained assistant turn; ``None`` if not prefix-stable."""
    spans: list[list[int]] = []
    for i, d in enumerate(dicts):
        if d.get("role") != "assistant" or d.get("train") is False:
            continue
        head = render(dicts[:i], tools, prompt=True)
        upto = render(dicts[: i + 1], tools)
        if not (upto.startswith(head) and text.startswith(upto)):
            return None
        if len(upto) > len(head):
            spans.append([len(head), len(upto)])
    return spans
