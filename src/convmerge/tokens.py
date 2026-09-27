"""Token lengths and chat-template checks for a training file (``convmerge tokens``).

Every row is read the way ``convert`` reads it (any supported layout:
``messages``, ``preference`` pairs, ShareGPT, Alpaca, ...), rendered with the
target model's chat template exactly as a trainer would render it, and
tokenized. That gives:

- the token-length distribution (percentiles and a histogram) and how many
  rows exceed ``max_tokens``;
- every row the chat template **rejects** (for example roles that do not
  alternate, a system turn the template does not allow, tool calls it
  cannot render), grouped by the template's error message — problems that
  otherwise surface only after training has started;
- rows whose tool-call arguments are stored as JSON strings but that the
  template encodes a second time (it expects objects — convert with
  ``--tool-arguments object``);
- optionally, a filtered copy: rows that render and fit go to ``output``,
  the rest to ``rejects``, both byte-for-byte as read.

A preference row counts as long as its longer side (prompt + chosen or
prompt + rejected). Needs ``pip install "convmerge[tokens]"`` (``transformers``;
no PyTorch).
"""

from __future__ import annotations

import json
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from convmerge.io import ReadStats, iter_jsonl

_BATCH = 256
_REASON_CHARS = 160


@dataclass
class TokenStats:
    """Counters filled by :func:`check_tokens`; ``to_report()`` gives the JSON report."""

    rows: int = 0
    measured: int = 0
    unreadable: int = 0
    """Rows no adapter could read as a conversation."""
    template_errors: dict[str, int] = field(default_factory=dict)
    error_lines: dict[str, list[int]] = field(default_factory=dict)
    over_limit: int = 0
    double_encoded_arguments: int = 0
    """Rows whose tool-call arguments are stored as JSON strings and that the
    chat template encodes again (it expects objects: ``--tool-arguments object``)."""
    kept: int = 0
    rejected: int = 0
    invalid_json: int = 0
    first_invalid_line: int | None = None
    max_tokens: int | None = None
    tokenizer: str | None = None
    lengths: array = field(default_factory=lambda: array("I"), repr=False)

    def to_report(self) -> dict[str, Any]:
        lengths = sorted(self.lengths)
        report: dict[str, Any] = {
            "tokenizer": self.tokenizer,
            "rows": self.rows,
            "measured": self.measured,
            "unreadable": self.unreadable,
            "invalid_json": self.invalid_json,
            "template_errors": dict(sorted(self.template_errors.items())),
            "error_lines": dict(sorted(self.error_lines.items())),
            "max_tokens": self.max_tokens,
            "over_limit": self.over_limit,
            "double_encoded_arguments": self.double_encoded_arguments,
            "kept": self.kept,
            "rejected": self.rejected,
        }
        if lengths:
            report["tokens"] = {
                "min": lengths[0],
                "mean": round(sum(lengths) / len(lengths), 1),
                **{f"p{p}": _percentile(lengths, p) for p in (50, 90, 95, 99)},
                "max": lengths[-1],
                "total": sum(lengths),
            }
            report["histogram"] = _histogram(lengths)
        return report


def load_tokenizer(
    name_or_path: str, *, revision: str | None = None, token: str | None = None
) -> Any:
    """``transformers.AutoTokenizer.from_pretrained`` with a clear install hint."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "convmerge tokens needs transformers: pip install 'convmerge[tokens]' (or [all])"
        ) from e
    return AutoTokenizer.from_pretrained(name_or_path, revision=revision, token=token)


def check_tokens(
    path: str | Path,
    *,
    tokenizer: Any,
    max_tokens: int | None = None,
    output: str | Path | None = None,
    rejects: str | Path | None = None,
    chat_template: str | None = None,
    encoding: str = "utf-8",
    stats: TokenStats | None = None,
) -> TokenStats:
    """Measure (and with ``output``, filter) ``path`` for ``tokenizer``.

    ``tokenizer`` is a model name or path (loaded with :func:`load_tokenizer`)
    or an already loaded ``transformers`` tokenizer. ``chat_template``
    overrides the tokenizer's own template (Jinja source). With ``output``,
    rows that render and are at most ``max_tokens`` long are written there
    and the others to ``rejects`` (if given).
    """
    tok = load_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
    template = chat_template if chat_template is not None else getattr(tok, "chat_template", None)
    if not template:
        raise ValueError(
            "the tokenizer has no chat template; pass one with --chat-template (a Jinja file)"
        )
    st = stats if stats is not None else TokenStats()
    st.max_tokens = max_tokens
    st.tokenizer = tokenizer if isinstance(tokenizer, str) else getattr(tok, "name_or_path", None)

    out = open(output, "w", encoding=encoding) if output is not None else None
    rej = open(rejects, "w", encoding=encoding) if rejects is not None else None
    try:
        batch: list[tuple[int, str, list[str]]] = []
        for number, raw, texts in _render(path, tok, template, encoding, st):
            if texts is None:  # unreadable or rejected by the template
                _write(rej, raw)
                st.rejected += out is not None
                continue
            batch.append((number, raw, texts))
            if len(batch) >= _BATCH:
                _measure(batch, tok, max_tokens, st, out, rej)
                batch = []
        _measure(batch, tok, max_tokens, st, out, rej)
    finally:
        for f in (out, rej):
            if f is not None:
                f.close()
    return st


def _render(
    path: str | Path, tok: Any, template: str, encoding: str, st: TokenStats
) -> Iterator[tuple[int, str, list[str] | None]]:
    from convmerge.adapter_resolve import resolve_adapter
    from convmerge.emitters import ToolArguments, _message_dict, split_pair

    adapter = resolve_adapter("auto", None, pairs=True)
    read = ReadStats()
    try:
        for line in iter_jsonl(path, encoding=encoding, stats=read):
            st.rows += 1
            examples = list(adapter(line.value)) if isinstance(line.value, dict) else []
            # Render tool-call arguments the way the file stores them, as a trainer would.
            as_strings = _stores_string_arguments(line.value)
            args: ToolArguments = "string" if as_strings else "object"
            double_encoded = False
            if not examples:
                st.unreadable += 1
                yield line.number, line.raw, None
                continue
            texts: list[str] = []
            error: str | None = None
            for ex in examples:
                if ex.rejected is not None:
                    try:
                        prompt, chosen, rejected = split_pair(ex)
                        sides = [prompt + chosen, prompt + rejected]
                    except ValueError:
                        sides = [ex.messages]
                else:
                    sides = [ex.messages]
                for msgs in sides:
                    dicts = [_message_dict(m, args) for m in msgs]
                    try:
                        text = tok.apply_chat_template(
                            dicts, tools=ex.tools, chat_template=template, tokenize=False
                        )
                    except Exception as e:  # noqa: BLE001 - templates raise anything
                        error = f"{type(e).__name__}: {e}"[:_REASON_CHARS]
                        break
                    texts.append(text)
                    if as_strings and not double_encoded:
                        double_encoded = any(
                            json.dumps(tc.arguments) in text for m in msgs for tc in m.tool_calls
                        )
                if error:
                    break
            st.double_encoded_arguments += double_encoded
            if error:
                st.template_errors[error] = st.template_errors.get(error, 0) + 1
                lines = st.error_lines.setdefault(error, [])
                if len(lines) < 5:
                    lines.append(line.number)
                yield line.number, line.raw, None
            else:
                yield line.number, line.raw, texts
    finally:
        st.invalid_json = read.invalid_json
        st.first_invalid_line = read.first_invalid_line


def _stores_string_arguments(value: Any) -> bool:
    """Whether an OpenAI-style row keeps tool-call arguments as JSON strings."""
    if not isinstance(value, dict):
        return False
    for key in ("messages", "prompt", "chosen", "rejected"):
        turns = value.get(key)
        if not isinstance(turns, list):
            continue
        for turn in turns:
            calls = turn.get("tool_calls") if isinstance(turn, dict) else None
            for call in calls if isinstance(calls, list) else ():
                fn = call.get("function") if isinstance(call, dict) else None
                if isinstance(fn, dict) and "arguments" in fn:
                    return isinstance(fn["arguments"], str)
    return False


def _measure(
    batch: list[tuple[int, str, list[str]]],
    tok: Any,
    max_tokens: int | None,
    st: TokenStats,
    out: Any,
    rej: Any,
) -> None:
    if not batch:
        return
    flat = [t for _, _, texts in batch for t in texts]
    ids = tok(flat, add_special_tokens=False)["input_ids"]
    pos = 0
    for _, raw, texts in batch:
        length = max(len(ids[pos + i]) for i in range(len(texts)))
        pos += len(texts)
        st.measured += 1
        st.lengths.append(length)
        if max_tokens is not None and length > max_tokens:
            st.over_limit += 1
            if out is not None:
                st.rejected += 1
                _write(rej, raw)
            continue
        if out is not None:
            st.kept += 1
            _write(out, raw)


def _write(f: Any, raw: str) -> None:
    if f is not None:
        f.write(raw + "\n")


def _percentile(sorted_values: list[int], p: int) -> int:
    k = max(0, min(len(sorted_values) - 1, round(p / 100 * (len(sorted_values) - 1))))
    return sorted_values[k]


def _histogram(sorted_values: list[int]) -> dict[str, int]:
    """Counts per power-of-two bucket: ``"<256"``, ``"256-511"``, ``"512-1023"``, ..."""
    buckets: dict[str, int] = {}
    for v in sorted_values:
        if v < 256:
            label = "<256"
        else:
            low = 1 << (v.bit_length() - 1)
            label = f"{low}-{2 * low - 1}"
        buckets[label] = buckets.get(label, 0) + 1
    return buckets
