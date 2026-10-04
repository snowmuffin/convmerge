"""Rule-based quality filter for training files (``convmerge filter``).

Every row is read the way ``convert`` reads it (``messages``, ShareGPT,
Alpaca, preference pairs, ...) and checked against a set of rules. Rows that
break a rule are dropped (to ``rejects`` when given); the others are written
byte-for-byte as read. The rules are deterministic and need no model:

- ``empty_answer`` (on): the final answer is empty or shorter than
  ``min_answer_chars``.
- ``refusal`` (on): an answer is a refusal or an "as an AI language model"
  disclaimer (built-in English and Korean phrases, plus your own). In a
  conversation with a system prompt or tools, only disclaimers count: such
  an assistant declines out-of-scope requests on purpose.
- ``repetition`` (on): an answer or reasoning trace loops: at least
  ``repetition_max`` of its word ``repetition_ngram``-grams repeat an earlier
  one.
- ``near_identical_pair`` / ``rejected_empty`` (on, preference pairs): the two
  answers are the same once case and whitespace are ignored, or the rejected
  answer is empty.
- ``length`` (off): the answer text is shorter than ``min_chars`` or longer
  than ``max_chars``.
- ``slop`` (off): answers use ``slop_max`` or more stock phrases ("delve
  into", "it's important to note", ...).
- ``script`` (off): less than the given share of the conversation's letters
  are in a script (``{"hangul": 0.3}`` catches half-translated rows); code is
  ignored.
- ``patterns``: your own regular expressions, one rule each.

For preference pairs the rules read the chosen answer (a refused or looping
*rejected* answer is what a pair is for), and the report adds length-bias
statistics: how often the chosen answer is the longer one.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from convmerge import _parallel
from convmerge._text import count, excerpt, fold, ngrams, without_code, words
from convmerge.io import ReadStats, iter_jsonl, refuse_overwrite
from convmerge.models import ChatMessage, TrainingExample

RULES: dict[str, str] = {
    "empty_answer": "the final answer is empty or too short",
    "refusal": "an answer refuses or is an 'as an AI language model' disclaimer",
    "repetition": "an answer or reasoning trace repeats itself",
    "near_identical_pair": "the chosen and rejected answers differ only in case or whitespace",
    "rejected_empty": "the rejected answer of a pair is empty",
    "length": "the answer text is outside min_chars / max_chars",
    "slop": "answers use slop_max or more stock phrases",
    "script": "too few of the letters are in the required script",
}
DEFAULT_RULES: tuple[str, ...] = (
    "empty_answer",
    "refusal",
    "repetition",
    "near_identical_pair",
    "rejected_empty",
)

# Refusals and AI disclaimers, matched in case-folded text. ``_REFUSAL_ANYWHERE``
# phrases count anywhere in an answer; ``_REFUSAL_START`` phrases only near
# its start, opening the answer or a sentence ("I'm sorry, but I can't ..."),
# so answers that merely discuss what cannot be done are kept.
_REFUSAL_ANYWHERE = (
    "as an ai language model",
    "as a language model, i",
    "as a large language model",
    "as an ai model, i",
    "i am an ai language model",
    "i'm an ai language model",
    "i am just an ai",
    "i'm just an ai",
    "ai 언어 모델로서",
    "인공지능 언어 모델로서",
    "ai 언어 모델이기 때문에",
    "저는 ai 언어 모델",
    "저는 인공지능 언어 모델",
)
_REFUSAL_START = (
    "i'm sorry, but i can",
    "i am sorry, but i can",
    "i'm sorry, i can't",
    "i'm sorry, i cannot",
    "sorry, but i can't",
    "sorry, i can't",
    "i apologize, but i can",
    "i apologize, but i cannot",
    "i cannot fulfill",
    "i can't fulfill",
    "i cannot assist",
    "i can't assist",
    "i cannot help with",
    "i can't help with",
    "i cannot comply",
    "i can't comply",
    "i must decline",
    "i'm not able to help",
    "i am not able to help",
    "i'm unable to help",
    "i am unable to help",
    "i'm unable to assist",
    "i am unable to assist",
    "i'm unable to provide",
    "i am unable to provide",
    "죄송하지만",
    "죄송합니다만",
    "죄송합니다. 저는",
    "죄송합니다, 저는",
    "도와드릴 수 없습니다",
    "도와 드릴 수 없습니다",
    "답변드릴 수 없습니다",
    "답변을 드릴 수 없습니다",
    "요청을 들어드릴 수 없",
)
_START_CHARS = 200

_SLOP = (
    "delve into",
    "delving into",
    "a testament to",
    "rich tapestry",
    "tapestry of",
    "in the realm of",
    "navigating the complexities",
    "it's important to note that",
    "it is important to note that",
    "it's worth noting that",
    "it is worth noting that",
    "in today's fast-paced world",
    "in today's digital age",
    "ever-evolving landscape",
    "ever-changing landscape",
    "a myriad of",
    "a plethora of",
    "embark on a journey",
    "unlock the potential",
    "unleash the power",
    "seamlessly integrate",
    "shivers down",
    "barely above a whisper",
    "symphony of",
    "beacon of",
    "i hope this helps",
    "feel free to ask",
    "let me know if you have any other questions",
    "도움이 되셨길 바랍니다",
    "도움이 되었기를 바랍니다",
    "도움이 되었으면 좋겠습니다",
    "궁금한 점이 있으시면 언제든지",
    "추가 질문이 있으시면",
    "결론적으로,",
    "종합적으로 볼 때",
)

_SCRIPTS: dict[str, re.Pattern[str]] = {
    "hangul": re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]"),
    "latin": re.compile(r"[A-Za-zÀ-ɏ]"),
    "han": re.compile(r"[㐀-䶿一-鿿豈-﫿]"),
    "kana": re.compile(r"[぀-ヿ]"),
    "cyrillic": re.compile(r"[Ѐ-ӿ]"),
}
_MIN_SCRIPT_LETTERS = 20
_MIN_REPETITION_TOKENS = 50
_SAMPLES = 3


@dataclass(frozen=True)
class FilterSpec:
    """Which rules :func:`filter_jsonl` applies, and their thresholds."""

    rules: tuple[str, ...] = DEFAULT_RULES
    min_answer_chars: int = 1
    min_chars: int | None = None
    max_chars: int | None = None
    repetition_ngram: int = 10
    repetition_max: float = 0.7
    slop_max: int = 3
    min_script: Mapping[str, float] | None = None
    refusal_phrases: tuple[str, ...] = ()
    """Extra refusal phrases, matched anywhere in an answer (case-insensitive)."""
    slop_phrases: tuple[str, ...] = ()
    builtin_phrases: bool = True
    """``False``: use only ``refusal_phrases`` / ``slop_phrases``."""
    patterns: Mapping[str, str] | None = None
    """Your own rules: name -> regular expression searched in the answers."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(dict.fromkeys(self.rules)))
        unknown = [r for r in self.rules if r not in RULES]
        if unknown:
            raise ValueError(f"unknown filter rule(s): {', '.join(unknown)} "
                             f"(known: {', '.join(RULES)})")  # fmt: skip
        if self.min_answer_chars < 0:
            raise ValueError("min_answer_chars: expected a non-negative integer")
        for name in ("min_chars", "max_chars"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name}: expected a non-negative integer")
        if self.repetition_ngram < 1:
            raise ValueError("repetition_ngram: expected a positive integer")
        if not 0 < self.repetition_max <= 1:
            raise ValueError("repetition_max: expected a fraction in (0, 1]")
        if self.slop_max < 1:
            raise ValueError("slop_max: expected a positive integer")
        for script, share in (self.min_script or {}).items():
            if script not in _SCRIPTS:
                raise ValueError(f"min_script: unknown script {script!r} "
                                 f"(known: {', '.join(_SCRIPTS)})")  # fmt: skip
            if not 0 < share <= 1:
                raise ValueError(f"min_script.{script}: expected a fraction in (0, 1]")
        if self.min_chars is not None and self.max_chars is not None:
            if self.min_chars > self.max_chars:
                raise ValueError(f"min_chars {self.min_chars} is greater than max_chars "
                                 f"{self.max_chars}: every row would be rejected")  # fmt: skip
        if "length" in self.rules and self.min_chars is None and self.max_chars is None:
            raise ValueError("the length rule needs min_chars or max_chars")
        if "script" in self.rules and not self.min_script:
            raise ValueError("the script rule needs min_script, e.g. {'hangul': 0.3}")
        for name, pattern in (self.patterns or {}).items():
            if name in RULES:
                raise ValueError(f"patterns.{name}: the name of a built-in rule")
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"patterns.{name}: invalid regular expression: {e}") from None

    @classmethod
    def from_options(
        cls,
        *,
        enable: Iterable[str] = (),
        disable: Iterable[str] = (),
        min_chars: int | None = None,
        max_chars: int | None = None,
        min_script: Mapping[str, float] | None = None,
        rules_file: str | Path | None = None,
        **options: Any,
    ) -> FilterSpec:
        """The default rules plus ``enable``, minus ``disable``.

        ``min_chars`` / ``max_chars`` turn ``length`` on and ``min_script``
        turns ``script`` on; ``rules_file`` adds phrases and patterns (see
        :func:`load_rules_file`). Other options are passed through.
        """
        off = set(disable)
        rules = [*DEFAULT_RULES, *enable]
        if min_chars is not None or max_chars is not None:
            rules.append("length")
        if min_script:
            rules.append("script")
        extra = load_rules_file(rules_file) if rules_file is not None else {}
        return cls(
            rules=tuple(r for r in rules if r not in off), min_chars=min_chars,
            max_chars=max_chars, min_script=dict(min_script) if min_script else None,
            **options, **extra,
        )  # fmt: skip

    @property
    def active(self) -> tuple[str, ...]:
        """The rule names this spec applies, in report order."""
        return (*self.rules, *(self.patterns or {}))


@dataclass
class FilterStats:
    """Counters filled by :func:`filter_jsonl`; ``to_report()`` gives the JSON report."""

    rows: int = 0
    kept: int = 0
    rejected: int = 0
    unreadable: int = 0
    invalid_json: int = 0
    first_invalid_line: int | None = None
    rules: dict[str, int] = field(default_factory=dict)
    """Rows each rule matched (a row can match several)."""
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    pairs: int = 0
    chosen_longer: int = 0
    length_ratios: list[float] = field(default_factory=list, repr=False)

    def to_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "rows": self.rows,
            "kept": self.kept,
            "rejected": self.rejected,
            "unreadable": self.unreadable,
            "invalid_json": self.invalid_json,
            "rules": dict(self.rules),
            "samples": {k: list(v) for k, v in self.samples.items() if v},
        }
        if self.pairs:
            report["preference"] = self.preference_report()
        return report

    def preference_report(self) -> dict[str, Any]:
        share = self.chosen_longer / self.pairs if self.pairs else 0.0
        ratios = self.length_ratios
        return {
            "pairs": self.pairs,
            "chosen_longer": self.chosen_longer,
            "chosen_longer_share": round(share, 4),
            "median_length_ratio": round(statistics.median(ratios), 4) if ratios else None,
        }

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.pairs >= 20 and self.chosen_longer / self.pairs > 0.7:
            share = self.chosen_longer / self.pairs
            out.append(
                f"the chosen answer is the longer one in {share:.0%} of "
                f"{count(self.pairs, 'pair')}; "
                "DPO on such data tends to learn length rather than quality"
            )
        return out


def filter_jsonl(
    path: str | Path,
    *,
    spec: FilterSpec | None = None,
    output: str | Path | None = None,
    rejects: str | Path | None = None,
    encoding: str = "utf-8",
    stats: FilterStats | None = None,
    workers: int = 1,
) -> FilterStats:
    """Check every row of ``path`` against ``spec``'s rules.

    With ``output``, rows that pass are written there and the others to
    ``rejects`` (if given), both byte-for-byte as read. Without it only the
    statistics are collected. ``workers`` > 1 checks rows in that many
    processes; the output and statistics are the same as with one.
    """
    refuse_overwrite([path], [output, rejects])
    if workers < 1:
        raise ValueError("workers: expected a positive integer")
    spec = spec or FilterSpec()
    st = stats if stats is not None else FilterStats()
    st.rules = {name: 0 for name in spec.active}
    for target in (output, rejects):
        if target is not None:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
    out = open(output, "w", encoding="utf-8") if output is not None else None
    rej = open(rejects, "w", encoding="utf-8") if rejects is not None else None
    try:
        if workers > 1:
            _filter_parallel(path, spec, st, out, rej, encoding, workers)
        else:
            _filter_serial(path, spec, st, out, rej, encoding)
    finally:
        for f in (out, rej):
            if f is not None:
                f.close()
    return st


def _filter_serial(
    path: str | Path, spec: FilterSpec, st: FilterStats, out: Any, rej: Any, encoding: str
) -> None:
    from convmerge.adapter_resolve import resolve_adapter

    checker = _Checker(spec)
    adapter = resolve_adapter("auto", None, pairs=True)
    read = ReadStats()
    try:
        for line in iter_jsonl(path, encoding=encoding, stats=read):
            f = out if _keep(line.value, line.number, adapter, checker, st) else rej
            if f is not None:
                _write(f, line.raw)
    finally:
        st.invalid_json = read.invalid_json
        st.first_invalid_line = read.first_invalid_line


def _keep(value: Any, number: int, adapter: Any, checker: _Checker, st: FilterStats) -> bool:
    """Check one row, counting it in ``st``; ``True`` when it passes."""
    st.rows += 1
    examples = list(adapter(value)) if isinstance(value, dict) else []
    usable = [ex for ex in examples if ex.messages]
    if not usable:
        st.unreadable += 1
        hits: dict[str, str] = {"unreadable": ""}
    else:
        hits = {}
        for ex in usable:
            for rule, text in checker.check(ex, st).items():
                hits.setdefault(rule, text)
        for rule, text in hits.items():
            st.rules[rule] += 1
            samples = st.samples.setdefault(rule, [])
            if len(samples) < _SAMPLES:
                samples.append({"line": number, "text": text})
    if hits:
        st.rejected += 1
        return False
    st.kept += 1
    return True


# --- filter --workers ---------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(spec: FilterSpec, encoding: str) -> None:
    from convmerge.adapter_resolve import resolve_adapter

    _WORKER.update(
        spec=spec,
        checker=_Checker(spec),
        adapter=resolve_adapter("auto", None, pairs=True),
        encoding=encoding,
    )


def _worker_chunk(chunk: _parallel.Chunk) -> tuple[str, str, FilterStats]:
    st = FilterStats(rules={name: 0 for name in _WORKER["spec"].active})
    kept: list[str] = []
    rejected: list[str] = []
    for number, raw, value in _parallel.parse_chunk(chunk, _WORKER["encoding"], st):
        keep = _keep(value, number, _WORKER["adapter"], _WORKER["checker"], st)
        (kept if keep else rejected).append(raw + "\n")
    return "".join(kept), "".join(rejected), st


def _filter_parallel(
    path: str | Path,
    spec: FilterSpec,
    st: FilterStats,
    out: Any,
    rej: Any,
    encoding: str,
    workers: int,
) -> None:
    chunks = _parallel.raw_chunks(path, encoding, ReadStats())
    parts = _parallel.ordered_map(
        chunks, _worker_chunk, workers=workers, initializer=_worker_init,
        initargs=(spec, encoding),
    )  # fmt: skip
    for kept, rejected, part in parts:
        if out is not None:
            out.write(kept)
        if rej is not None:
            rej.write(rejected)
        _merge(st, part)


def _merge(st: FilterStats, part: FilterStats) -> None:
    """Add a later chunk's statistics to ``st``."""
    for name in ("rows", "kept", "rejected", "unreadable", "pairs", "chosen_longer"):
        setattr(st, name, getattr(st, name) + getattr(part, name))
    _parallel.merge_invalid(st, part)
    for rule, n in part.rules.items():
        st.rules[rule] = st.rules.get(rule, 0) + n
    for rule, samples in part.samples.items():
        mine = st.samples.setdefault(rule, [])
        mine.extend(samples[: max(0, _SAMPLES - len(mine))])
    st.length_ratios.extend(part.length_ratios)


def _write(f: Any, raw: str) -> None:
    f.write(raw)
    f.write("\n")


class _Checker:
    def __init__(self, spec: FilterSpec) -> None:
        self.spec = spec
        builtin = spec.builtin_phrases
        self.anywhere = [fold(p) for p in (*(_REFUSAL_ANYWHERE if builtin else ()),
                                           *spec.refusal_phrases)]  # fmt: skip
        # A refusal opens the answer or a sentence near its start.
        self.start = (
            re.compile(r"(?:^|[.!?\n] ?)[\"'*]*("
                       + "|".join(re.escape(fold(p)) for p in _REFUSAL_START) + ")")
            if builtin
            else None
        )  # fmt: skip
        self.slop = [fold(p) for p in (*(_SLOP if builtin else ()), *spec.slop_phrases)]
        self.patterns = {name: re.compile(p) for name, p in (spec.patterns or {}).items()}

    def check(self, ex: TrainingExample, st: FilterStats) -> dict[str, str]:
        """Rule name -> an excerpt showing why, for each rule ``ex`` breaks."""
        rules = self.spec.rules
        hits: dict[str, str] = {}
        prompt: list[ChatMessage] = []
        answer = ex.messages
        if ex.rejected is not None:
            prompt, answer, rejected = _split(ex)
            st.pairs += 1
            chosen_len, rejected_len = _chars(answer), _chars(rejected)
            st.chosen_longer += chosen_len > rejected_len
            if rejected_len:
                st.length_ratios.append(chosen_len / rejected_len)
            if "near_identical_pair" in rules and _same(answer, rejected):
                hits["near_identical_pair"] = excerpt(_joined(answer))
            if "rejected_empty" in rules and not _has_answer(rejected):
                hits["rejected_empty"] = ""
        answers = [m for m in answer if m.role == "assistant"]
        texts = [m.text for m in answers]

        if "empty_answer" in rules:
            final = answers[-1] if answers else None
            if final is None or (
                not final.tool_calls and len(final.text.strip()) < self.spec.min_answer_chars
            ):
                hits["empty_answer"] = excerpt(final.text) if final else ""
        if "refusal" in rules:
            # An assistant scoped by a system prompt or tools declines requests
            # outside that scope on purpose; only disclaimers count there.
            scoped = bool(ex.tools) or any(m.role == "system" for m in ex.messages)
            found = self._refusal(texts, openings=not scoped)
            if found is not None:
                hits["refusal"] = found
        if "repetition" in rules:
            for text in (*texts, *(m.reasoning or "" for m in answers)):
                share = self._repeated_share(text)
                if share >= self.spec.repetition_max:
                    hits["repetition"] = f"{share:.0%} repeated: {excerpt(text)}"
                    break
        if "length" in rules:
            n = sum(len(t) for t in texts)
            lo, hi = self.spec.min_chars, self.spec.max_chars
            if (lo is not None and n < lo) or (hi is not None and n > hi):
                hits["length"] = f"{n} characters"
        if "slop" in rules:
            joined = fold("\n".join(texts))
            found_slop = [p for p in self.slop if p in joined]
            count = sum(joined.count(p) for p in found_slop)
            if count >= self.spec.slop_max:
                hits["slop"] = ", ".join(found_slop[:5])
        if "script" in rules:
            conversation = [m.text for m in (*prompt, *answer) if m.role != "system"]
            low = _script_shortfall(without_code("\n".join(conversation)),
                                    self.spec.min_script or {})  # fmt: skip
            if low is not None:
                hits["script"] = low
        for name, pattern in self.patterns.items():
            for text in texts:
                m = pattern.search(text)
                if m:
                    hits[name] = excerpt(text, m.start())
                    break
        return hits

    def _refusal(self, texts: Iterable[str], *, openings: bool = True) -> str | None:
        for text in texts:
            folded = fold(text)
            at = next((i for i in (folded.find(p) for p in self.anywhere) if i >= 0), -1)
            if at < 0 and openings and self.start is not None:
                m = self.start.search(folded[:_START_CHARS])
                at = m.start(1) if m else -1
            if at >= 0:
                # Show the original text where case folding kept the length.
                shown = fold(text, casefold=False)
                return excerpt(shown if len(shown) == len(folded) else folded, at)
        return None

    def _repeated_share(self, text: str) -> float:
        """Share of the word n-grams of ``text`` that repeat an earlier one."""
        tokens = words(text)
        if len(tokens) < max(_MIN_REPETITION_TOKENS, 2 * self.spec.repetition_ngram):
            return 0.0
        grams = list(ngrams(tokens, self.spec.repetition_ngram))
        return (len(grams) - len(set(grams))) / len(grams)


def _split(
    ex: TrainingExample,
) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage]]:
    """(prompt, chosen continuation, rejected continuation) without raising."""
    chosen, rejected = ex.messages, ex.rejected or []
    n = 0
    while n < len(chosen) and n < len(rejected) and chosen[n] == rejected[n]:
        n += 1
    return chosen[:n], chosen[n:], rejected[n:]


def _has_answer(turns: list[ChatMessage]) -> bool:
    return any(m.role == "assistant" and (m.text.strip() or m.tool_calls) for m in turns)


def _joined(turns: list[ChatMessage]) -> str:
    return "\n".join(m.text for m in turns if m.role == "assistant")


def _chars(turns: list[ChatMessage]) -> int:
    return len(_joined(turns))


def _same(a: list[ChatMessage], b: list[ChatMessage]) -> bool:
    if [m.tool_calls for m in a] != [m.tool_calls for m in b]:
        return False
    return fold(_joined(a)) == fold(_joined(b))


def _script_shortfall(text: str, minimum: Mapping[str, float]) -> str | None:
    letters = sum(ch.isalpha() for ch in text)
    if letters < _MIN_SCRIPT_LETTERS:
        return None
    for script, share in minimum.items():
        have = len(_SCRIPTS[script].findall(text)) / letters
        if have < share:
            return f"{script} {have:.0%} of {letters} letters (< {share:.0%})"
    return None


def load_rules_file(path: str | Path) -> dict[str, Any]:
    """Read a rules file (YAML or JSON) into :class:`FilterSpec` keyword arguments.

    Keys: ``refusal`` / ``slop`` (extra phrases), ``builtin_phrases`` (bool),
    ``patterns`` (name -> regular expression).
    """
    import json

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "YAML rules files need PyYAML: pip install 'convmerge[preset]', "
                "or write the file as JSON"
            ) from e
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping")
    unknown = set(raw) - {"refusal", "slop", "builtin_phrases", "patterns"}
    if unknown:
        raise ValueError(f"{path}: unknown key(s): {', '.join(sorted(unknown))} "
                         "(expected refusal, slop, builtin_phrases, patterns)")  # fmt: skip
    kwargs: dict[str, Any] = {}
    for key, target in (("refusal", "refusal_phrases"), ("slop", "slop_phrases")):
        phrases = raw.get(key, [])
        if not (isinstance(phrases, list) and all(isinstance(x, str) and x for x in phrases)):
            raise ValueError(f"{path}: {key}: expected a list of phrases")
        kwargs[target] = tuple(phrases)
    builtin = raw.get("builtin_phrases", True)
    if not isinstance(builtin, bool):
        raise ValueError(f"{path}: builtin_phrases: expected true or false")
    kwargs["builtin_phrases"] = builtin
    patterns = raw.get("patterns", {})
    if not (
        isinstance(patterns, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in patterns.items())
    ):
        raise ValueError(f"{path}: patterns: expected a mapping of name -> regular expression")
    kwargs["patterns"] = patterns
    return kwargs
