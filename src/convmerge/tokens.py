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
- whether the template marks assistant text with ``{% generation %}`` (TRL's
  ``assistant_only_loss`` needs it) and, with ``max_tokens``, rows whose
  first answer starts beyond the limit (truncated to that length they train
  on no answer at all);
- rows whose rendered answer is not followed by a stop token (the
  tokenizer's ``eos_token`` or a ``generation_config.json`` ``eos_token_id``),
  so the model never learns to stop;
- rows whose reasoning trace does not appear in the rendered text (most
  reasoning templates render it only after the last user turn, and each
  reads its own field: ``reasoning_content`` or ``thinking``);
- the ``instruction_part`` / ``response_part`` strings for Unsloth's
  ``train_on_responses_only`` (the text the template puts before a user and
  before an assistant turn), and rows in which they do not appear as tokens
  (Unsloth would mask those rows entirely);
- ``hints``: the ``convert`` options that fix what was found;
- optionally, a filtered copy: rows that render and fit go to ``output``,
  the rest to ``rejects``, both byte-for-byte as read.

A preference row counts as long as its longer side (prompt + chosen or
prompt + rejected). Needs ``pip install "convmerge[tokens]"`` (``transformers``;
no PyTorch).
"""

from __future__ import annotations

import json
import re
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from convmerge._text import agree, count
from convmerge.io import ReadStats, iter_jsonl, refuse_overwrite

_BATCH = 256
_REASON_CHARS = 160
_SNIPPET = 40
_GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")


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
    generation_tags: bool | None = None
    """Whether the chat template marks assistant text with ``{% generation %}``."""
    answer_beyond_limit: int = 0
    """Rows whose first answer starts at or after ``max_tokens`` (only with a limit)."""
    stop_tokens: list[str] = field(default_factory=list)
    missing_eos: int = 0
    """Rows whose rendered final answer is not followed by any of ``stop_tokens``."""
    reasoning_dropped: int = 0
    """Rows with a reasoning trace the rendered text does not contain."""
    reasoning_dropped_final: int = 0
    """Of those, rows where even the final answer's trace is missing."""
    lengths: array = field(default_factory=lambda: array("I"), repr=False)
    response_markers: dict[str, str] | None = None
    """``instruction_part`` / ``response_part`` for Unsloth's ``train_on_responses_only``
    (``None`` when the template's turn headers could not be worked out)."""
    markers_missing: int = 0
    """Rows whose tokens do not contain both markers' tokens."""

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
            "generation_tags": self.generation_tags,
            "answer_beyond_limit": self.answer_beyond_limit,
            "stop_tokens": list(self.stop_tokens),
            "missing_eos": self.missing_eos,
            "reasoning_dropped": self.reasoning_dropped,
            "reasoning_dropped_final": self.reasoning_dropped_final,
            "response_markers": self.response_markers,
            "markers_missing": self.markers_missing,
            "kept": self.kept,
            "rejected": self.rejected,
            "hints": self.hints(),
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

    def hints(self) -> list[str]:
        """The ``convert`` options (or settings) that fix what this check found."""
        out: list[str] = []
        errors = " ".join(self.template_errors).lower()
        if "alternate" in errors:
            out.append(
                "roles must alternate: convert with --merge-consecutive (and --system fold "
                "if a system turn is in the way)"
            )
        if "system role" in errors or "system message" in errors:
            out.append("the template has no system role: convert with --system fold")
        if "nonetype" in errors:
            out.append(
                "the template fails on null content: convert with --tool-content empty "
                "(the default since convmerge 0.12)"
            )
        if 'not "dict"' in errors or "'dict object'" in errors:
            out.append(
                "the template expects tool-call arguments as a JSON string: convert with "
                "--tool-arguments string"
            )
        if "'str object' has no attribute 'items'" in errors:
            out.append(
                "the template expects tool-call arguments as an object: convert with "
                "--tool-arguments object"
            )
        if self.double_encoded_arguments:
            out.append(
                "tool-call arguments are encoded twice: convert with --tool-arguments object"
            )
        if self.reasoning_dropped_final:
            out.append(
                "the template does not render the stored reasoning: convert with --reasoning "
                "reasoning_content (Qwen3, DeepSeek), --reasoning thinking (gpt-oss), or "
                "--reasoning inline"
            )
        if self.reasoning_dropped > self.reasoning_dropped_final:
            out.append(
                "the template leaves out earlier reasoning traces (most render reasoning "
                "only after the last user turn), so a trainer that applies it never trains "
                "on them: convert with --split-turns to train on every turn's trace, or "
                "--reasoning-turns last to drop them from the data"
            )
        if self.generation_tags is False:
            out.append(
                "the template has no {% generation %} markers: if you train with TRL "
                "assistant_only_loss, use a template that has them (otherwise the loss "
                "covers the prompt too, or nothing at all)"
            )
        if self.answer_beyond_limit:
            out.append(
                f"{count(self.answer_beyond_limit, 'row')} "
                f"{agree(self.answer_beyond_limit, 'starts its', 'start their')} answer beyond "
                f"{count(self.max_tokens or 0, 'token')} and "
                f"{agree(self.answer_beyond_limit, 'trains', 'train')} on nothing when "
                "truncated; filter them "
                "with -o (tokens --max-tokens) or raise the trainer's max length"
            )
        if self.markers_missing:
            out.append(
                f"{count(self.markers_missing, 'row')} "
                f"{agree(self.markers_missing, 'does', 'do')} not contain the response_markers "
                "as tokens: "
                "Unsloth train_on_responses_only would mask all of their labels; check the "
                "markers against the template the trainer uses"
            )
        if self.missing_eos:
            stops = ", ".join(self.stop_tokens) or "none"
            out.append(
                f"answers are not followed by a stop token ({stops}), so the model does not "
                "learn to stop: set the tokenizer's eos_token (or generation_config "
                "eos_token_id) to the template's end-of-turn token"
            )
        return out


def load_tokenizer(
    name_or_path: str, *, revision: str | None = None, token: str | None = None
) -> Any:
    """``transformers.AutoTokenizer.from_pretrained`` with a clear install hint."""
    _require_template_support()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path, revision=revision, token=token)


def _require_template_support() -> None:
    # transformers renders chat templates with jinja2 but does not depend on it.
    try:
        import jinja2  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"convmerge tokens needs transformers and jinja2 ({e.name} is missing): "
            "pip install 'convmerge[tokens]' (or [all])"
        ) from e


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
    from convmerge._paths import protect_paths

    refuse_overwrite([path], [output, rejects])
    name = tokenizer if isinstance(tokenizer, str) else getattr(tokenizer, "name_or_path", None)
    local = Path(name).expanduser() if name else None
    protect_paths(
        [path, local if local is not None and local.exists() else None], [output, rejects]
    )
    _require_template_support()
    tok = load_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
    template = chat_template if chat_template is not None else getattr(tok, "chat_template", None)
    if not template:
        raise ValueError(
            "the tokenizer has no chat template; pass one with --chat-template (a Jinja file)"
        )
    st = stats if stats is not None else TokenStats()
    st.max_tokens = max_tokens
    st.tokenizer = tokenizer if isinstance(tokenizer, str) else getattr(tok, "name_or_path", None)
    st.generation_tags = bool(_GENERATION_TAG.search(template))
    st.stop_tokens = _stop_tokens(tok)
    st.response_markers = response_markers(tok, template)
    marker_ids: list[list[int]] = []
    if st.response_markers:
        marker_ids = [
            tok(text, add_special_tokens=False)["input_ids"]
            for text in st.response_markers.values()
        ]

    out = open(output, "w", encoding="utf-8") if output is not None else None
    rej = open(rejects, "w", encoding="utf-8") if rejects is not None else None
    try:
        batch: list[_Row] = []
        for row in _render(path, tok, template, encoding, st, prefixes=max_tokens is not None):
            if row.texts is None:  # unreadable or rejected by the template
                _write(rej, row.raw)
                st.rejected += out is not None
                continue
            batch.append(row)
            if len(batch) >= _BATCH:
                _measure(batch, tok, max_tokens, st, out, rej, marker_ids)
                batch = []
        _measure(batch, tok, max_tokens, st, out, rej, marker_ids)
    finally:
        for f in (out, rej):
            if f is not None:
                f.close()
    return st


@dataclass
class _Row:
    number: int
    raw: str
    texts: list[str] | None
    prefix: str | None = None
    """The rendered prompt up to the first answer (with the generation prompt)."""


def _render(
    path: str | Path,
    tok: Any,
    template: str,
    encoding: str,
    st: TokenStats,
    *,
    prefixes: bool = False,
) -> Iterator[_Row]:
    from convmerge.adapter_resolve import resolve_adapter
    from convmerge.emitters import _message_dict, split_pair

    adapter = resolve_adapter("auto", None, pairs=True)
    read = ReadStats()
    try:
        for line in iter_jsonl(path, encoding=encoding, stats=read):
            st.rows += 1
            examples = list(adapter(line.value)) if isinstance(line.value, dict) else []
            # Native message rows must be checked as stored, not reconstructed
            # with one argument/reasoning type chosen for the whole row.
            stored = _stored_sides(line.value)
            if (
                len(examples) != 1
                or stored is not None
                and len(stored) != (2 if examples[0].rejected is not None else 1)
            ):
                stored = None
            key = _reasoning_key(line.value)
            side_index = 0
            double_encoded = missing_eos = dropped = dropped_final = False
            if not examples:
                st.unreadable += 1
                yield _Row(line.number, line.raw, None)
                continue
            texts: list[str] = []
            prefix: str | None = None
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
                    dicts = (
                        stored[side_index]
                        if stored is not None
                        else [
                            _message_dict(m, "object", reasoning_key=key, tool_content="null")
                            for m in msgs
                        ]
                    )
                    side_index += 1
                    tools = line.value.get("tools", ex.tools) if stored is not None else ex.tools
                    try:
                        text = tok.apply_chat_template(
                            dicts, tools=tools, chat_template=template, tokenize=False
                        )
                        if not isinstance(text, str):
                            raise TypeError("the template did not render text")
                    except ImportError:
                        raise  # a missing dependency, not a problem with this row
                    except Exception as e:  # noqa: BLE001 - templates raise anything
                        error = f"{type(e).__name__}: {e}"[:_REASON_CHARS]
                        break
                    texts.append(text)
                    if not double_encoded:
                        double_encoded = any(
                            json.dumps(value) in text for value in _argument_strings(dicts)
                        )
                    if prefixes and prefix is None:
                        prefix = _render_prefix(tok, template, dicts, tools)
                    if st.stop_tokens and not missing_eos:
                        missing_eos = _missing_stop(msgs, text, st.stop_tokens)
                    lost, lost_final = _lost_reasoning(msgs, text)
                    dropped, dropped_final = dropped or lost, dropped_final or lost_final
                if error:
                    break
            st.double_encoded_arguments += double_encoded
            if error:
                st.template_errors[error] = st.template_errors.get(error, 0) + 1
                lines = st.error_lines.setdefault(error, [])
                if len(lines) < 5:
                    lines.append(line.number)
                yield _Row(line.number, line.raw, None)
                continue
            st.missing_eos += missing_eos
            st.reasoning_dropped += dropped
            st.reasoning_dropped_final += dropped_final
            yield _Row(line.number, line.raw, texts, prefix)
    finally:
        st.invalid_json = read.invalid_json
        st.first_invalid_line = read.first_invalid_line


def _render_prefix(tok: Any, template: str, msgs: list[dict[str, Any]], tools: Any) -> str | None:
    """The unmodified prompt before the first answer, with the generation prompt."""
    first = next((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), None)
    if not first:
        return None
    dicts = msgs[:first]
    try:
        text = tok.apply_chat_template(
            dicts, tools=tools, chat_template=template, tokenize=False, add_generation_prompt=True
        )
    except Exception:  # noqa: BLE001 - some templates refuse a prompt-only conversation
        return None
    return text if isinstance(text, str) else None


def _missing_stop(msgs: list[Any], text: str, stops: list[str]) -> bool:
    """Whether the final answer's rendered text is not followed by a stop token."""
    if not msgs or msgs[-1].role != "assistant":
        return False
    answer = msgs[-1].text.strip()
    if not answer:
        return False
    snippet = answer[-_SNIPPET:]
    at = text.rfind(snippet)
    if at < 0:
        return False  # the template rewrote the answer; nothing to anchor on
    tail = text[at + len(snippet) :]
    return not any(stop in tail for stop in stops)


def _lost_reasoning(msgs: list[Any], text: str) -> tuple[bool, bool]:
    """(some reasoning trace is missing from ``text``, the final answer's is)."""
    from convmerge.reasoning import reasoning_text

    last = max((i for i, m in enumerate(msgs) if m.role == "assistant"), default=-1)
    lost = lost_final = False
    for i, m in enumerate(msgs):
        if m.role != "assistant":
            continue
        trace = (reasoning_text(m) or "").strip()
        if trace and trace[:_SNIPPET] not in text:
            lost = True
            lost_final = lost_final or i == last
    return lost, lost_final


def _reasoning_key(value: Any) -> str:
    """The turn key a row stores reasoning under (``thinking`` or ``reasoning_content``)."""
    if isinstance(value, dict):
        for key in ("messages", "prompt", "chosen", "rejected", "conversations"):
            turns = value.get(key)
            for turn in turns if isinstance(turns, list) else ():
                if isinstance(turn, dict) and isinstance(turn.get("thinking"), str):
                    return "thinking"
    return "reasoning_content"


def _stop_tokens(tok: Any) -> list[str]:
    """The tokenizer's ``eos_token`` plus the ``eos_token_id`` of ``generation_config.json``."""
    stops: list[str] = []
    eos = getattr(tok, "eos_token", None)
    if isinstance(eos, str) and eos:
        stops.append(eos)
    for token_id in _generation_eos_ids(getattr(tok, "name_or_path", None)):
        try:
            token = tok.convert_ids_to_tokens(token_id)
        except Exception:  # noqa: BLE001 - an id outside the vocabulary
            continue
        if isinstance(token, str) and token and token not in stops:
            stops.append(token)
    return stops


def _generation_eos_ids(name_or_path: str | None) -> list[int]:
    if not name_or_path:
        return []
    local = Path(name_or_path)
    try:
        if local.is_dir():
            path = local / "generation_config.json"
            if not path.is_file():
                return []
        else:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(name_or_path, "generation_config.json"))
        ids = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id")
    except Exception:  # noqa: BLE001 - no generation config is fine
        return []
    if isinstance(ids, int):
        return [ids]
    return [i for i in ids if isinstance(i, int)] if isinstance(ids, list) else []


def _stored_sides(value: Any) -> list[list[dict[str, Any]]] | None:
    """Native SFT/DPO messages, preserving each field's type and absence.

    Non-native layouts still use their adapter's canonical representation.
    This helper never rewrites an argument string, content, reasoning or train flag.
    """
    if not isinstance(value, dict):
        return None

    def native(turns: Any) -> bool:
        return isinstance(turns, list) and all(
            isinstance(m, dict)
            and isinstance(m.get("role"), str)
            and m["role"] in {"system", "developer", "user", "assistant", "tool", "function"}
            for m in turns
        )

    if "conversation_a" in value and "conversation_b" in value:
        return None  # the adapter gives these higher priority than messages
    if "chosen" in value and "rejected" in value:
        prompt = value.get("prompt", value.get("messages", []))
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}] if prompt else []
        if not native(prompt):
            return None
        sides = []
        for key in ("chosen", "rejected"):
            branch = value[key]
            if isinstance(branch, str):
                branch = [{"role": "assistant", "content": branch}]
            if not native(branch) or not branch:
                return None
            combined = branch if prompt and branch[: len(prompt)] == prompt else prompt + branch
            if not any(m["role"] == "user" for m in combined):
                return None
            sides.append(combined)
        return sides
    messages = value.get("messages")
    if (
        native(messages)
        and messages
        and messages[-1]["role"] in {"assistant", "tool", "function"}
        and any(m["role"] == "user" for m in messages)
    ):
        return [messages]
    return None


def _argument_strings(messages: list[dict[str, Any]]) -> Iterator[str]:
    for message in messages:
        calls = message.get("tool_calls")
        for call in calls if isinstance(calls, list) else ():
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                yield function["arguments"]


def _measure(
    batch: list[_Row],
    tok: Any,
    max_tokens: int | None,
    st: TokenStats,
    out: Any,
    rej: Any,
    marker_ids: list[list[int]] | None = None,
) -> None:
    if not batch:
        return
    flat: list[str] = []
    for row in batch:
        assert row.texts is not None
        flat.extend(row.texts)
        if row.prefix is not None:
            flat.append(row.prefix)
    ids = tok(flat, add_special_tokens=False)["input_ids"]
    pos = 0
    for row in batch:
        texts = row.texts or []
        raw = row.raw
        length = max(len(ids[pos + i]) for i in range(len(texts)))
        if (
            marker_ids
            and texts
            and any(
                not all(_contains(ids[pos + i], m) for m in marker_ids) for i in range(len(texts))
            )
        ):
            st.markers_missing += 1
        pos += len(texts)
        if row.prefix is not None:
            if max_tokens is not None and len(ids[pos]) >= max_tokens:
                st.answer_beyond_limit += 1
            pos += 1
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


_PROBE = ("QQQ1", "AAA1", "QQQ2")


def response_markers(tok: Any, template: str) -> dict[str, str] | None:
    """The text a chat template puts before a user turn and before an assistant turn.

    ``response_part`` is the generation prompt (what the template adds for
    the model to answer); ``instruction_part`` is what follows the end of an
    assistant turn before the next user message, minus the turn end the
    template also writes after a user turn. These are the strings Unsloth's
    ``train_on_responses_only(instruction_part=..., response_part=...)``
    expects. ``None`` when the template does not fit this pattern.
    """
    q1, a1, q2 = _PROBE

    def render(msgs: list[tuple[str, str]], gen: bool = False) -> str:
        dicts = [{"role": r, "content": c} for r, c in msgs]
        text = tok.apply_chat_template(dicts, chat_template=template, tokenize=False,
                                       add_generation_prompt=gen)  # fmt: skip
        if not isinstance(text, str):
            raise TypeError("the template did not render text")
        return text

    try:
        prompt = render([("user", q1)])
        with_gen = render([("user", q1)], gen=True)
        one = render([("user", q1), ("assistant", a1)])
        two = render([("user", q1), ("assistant", a1), ("user", q2)])
    except Exception:  # noqa: BLE001 - templates raise anything
        return None
    if not with_gen.startswith(prompt):
        return None
    response = with_gen[len(prompt) :]
    user_to_answer = one[one.find(q1) + len(q1) : one.find(a1)]
    answer_to_user = two[two.find(a1) + len(a1) : two.find(q2)]
    at = user_to_answer.find(response) if response.strip() else -1
    if at < 0:
        return None
    turn_end = user_to_answer[:at]
    if not answer_to_user.startswith(turn_end):
        return None
    instruction = answer_to_user[len(turn_end) :]
    if not instruction.strip() or response in instruction or instruction in response:
        return None  # not two distinct turn headers (e.g. a template that needs content parts)
    return {"instruction_part": instruction, "response_part": response}


def _contains(seq: list[int], sub: list[int]) -> bool:
    if not sub:
        return True
    hay, needle = array("I", seq).tobytes(), array("I", sub).tobytes()
    at = hay.find(needle)
    while at >= 0:
        if at % 4 == 0:
            return True
        at = hay.find(needle, at + 1)
    return False


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
