"""mix command: weighted sampling of converted JSONL sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from convmerge.cli._common import config_errors


def _add_mix(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "mix",
        help=(
            "Sample from multiple converted JSONL sources at specified weights "
            "and merge into one file (install convmerge[preset] for YAML configs)"
        ),
    )
    p.add_argument(
        "config",
        nargs="?",
        default=None,
        metavar="CONFIG",
        help="Path to a YAML or JSON mix config file",
    )
    p.add_argument(
        "--input",
        "-i",
        nargs="+",
        action="extend",
        metavar="FILE:WEIGHT",
        default=None,
        help="Inline sources as path:weight pairs, e.g. code.jsonl:0.4 math.jsonl:0.6",
    )
    p.add_argument("--output", "-o", type=Path, default=None, help="Output JSONL path")
    p.add_argument("--total", "-n", type=int, default=None, help="Target total record count")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: config 'seed', else 42)",
    )
    p.add_argument(
        "--oversample",
        action="store_true",
        help="Allow sampling with replacement when a source has fewer records than requested",
    )
    p.add_argument(
        "--sampler",
        choices=("v1", "v2"),
        default=None,
        help="v2 (default): streaming, bounded memory. v1: the 0.6 in-memory "
        "sampler, to reproduce a mix made with an earlier version",
    )
    p.add_argument(
        "--by",
        choices=("rows", "chars", "tokens"),
        default=None,
        help="what the weights measure (default rows). chars / tokens: --total rows are "
        "split so each source's share of the characters or tokens is its weight",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        metavar="NAME_OR_PATH",
        help="Hugging Face tokenizer for --by tokens (needs transformers)",
    )
    p.add_argument(
        "--by-sample",
        type=_positive_int,
        default=None,
        metavar="N",
        help="with --by chars / tokens: measure N random rows per source and scale "
        "(an estimate; much faster for tokens on large sources). Default: every row",
    )
    p.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=None,
        metavar="N",
        help="with --by tokens: count each row as at most N tokens, the trainer's "
        "maximum length, so the weights describe the text that is trained on",
    )
    p.add_argument(
        "--no-recipe",
        action="store_true",
        help="Skip writing the .mix.json sidecar file",
    )
    p.add_argument(
        "--encoding", default="utf-8",
        help="Encoding of the source files (default: utf-8); output is always UTF-8",
    )  # fmt: skip


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return n


def mix_summary(result, *, oversample: bool) -> list[str]:
    """Per-source lines, then the total; shares are estimated from each source's
    mean row length and the share of its rows with a reasoning trace."""
    lines = []
    written_units = [
        s.written * (s.mean_units or 0.0) for s in result.sources
    ]  # the expected size of each source's sample
    all_units = sum(written_units)
    traces = 0.0
    for s, units in zip(result.sources, written_units):
        clipped = s.available < s.requested and not oversample
        note = f" (clipped from {s.requested:,})" if clipped else ""
        extra = ""
        if result.by != "rows" and all_units:
            extra += f" {result.by}={units / all_units:.1%}"
        if s.over_cap and s.available:
            extra += f" over_{result.max_tokens}={s.over_cap / s.available:.0%}"
        if s.reasoning and s.available:
            share = s.reasoning / s.available
            traces += s.written * share
            extra += f" reasoning={share:.0%}"
        lines.append(f"  {s.path}: weight={s.weight:.4f} written={s.written:,}{extra}{note}")
    lines.append(f"total written: {result.total_written:,} -> {result.output}")
    sampled = [s for s in result.sources if s.measured is not None and s.measured < s.available]
    if sampled:
        lines.append(
            f"{result.by} per row estimated from {result.by_sample:,} random "
            f"rows of {len(sampled)} source(s)"
        )
    if result.max_tokens is not None:
        lines.append(
            f"tokens counted up to {result.max_tokens:,} per row; over_{result.max_tokens} "
            "is each source's share of rows longer than that, which the trainer cuts"
        )
    if traces and result.total_written:
        lines.append(
            f"rows with a reasoning trace: about {traces / result.total_written:.0%} "
            "(each source's share applied to its sample)"
        )
    return lines


def _option_problem(
    args: argparse.Namespace,
    *,
    by: str,
    total: int | None,
    sampler: str,
    tokenizer: str | None,
    by_sample: int | None,
    max_tokens: int | None,
) -> str | None:
    """The first invalid option combination, named as the user wrote it.

    :func:`convmerge.mix.mix_files` checks the same rules but names its own
    parameters (``by='tokens'``); here each option is named by its flag, or by
    its key when it came from the config file.
    """

    def name(dest: str, flag: str) -> str:
        return flag if getattr(args, dest) not in (None, False) else f"'{dest}' in the config"

    by_name = f"--by {by}" if args.by else f"'by: {by}' in the config"
    if by != "rows" and total is None:
        return (
            f"{by_name} needs a total (-n/--total N): without one every record is "
            "merged and the weights are not used"
        )
    if by != "rows" and sampler != "v2":
        return f"{by_name} needs the v2 sampler ({name('sampler', '--sampler')} is {sampler})"
    if by == "tokens" and tokenizer is None:
        return f"{by_name} needs --tokenizer NAME (the tokenizer that counts the tokens)"
    if by_sample is not None and by == "rows":
        return f"{name('by_sample', '--by-sample')} needs --by chars or --by tokens"
    if max_tokens is not None and by != "tokens":
        return f"{name('max_tokens', '--max-tokens')} needs --by tokens"
    if args.tokenizer and by != "tokens":
        return "--tokenizer needs --by tokens (the tokenizer only counts tokens for the weights)"
    return None


def _cmd_mix(args: argparse.Namespace) -> None:
    from convmerge.mix import MixSource, load_mix_config, mix_files, write_mix_recipe

    sources: list[MixSource] = []
    options: dict = {}

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            print(f"error: config file not found: {config_path}", file=sys.stderr)
            sys.exit(2)
        try:
            sources, options = load_mix_config(config_path)
        except config_errors() as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
    elif args.input:
        for spec in args.input:
            try:
                file_part, weight_part = spec.rsplit(":", 1)
                sources.append(MixSource(Path(file_part), float(weight_part)))
            except ValueError:
                print(
                    f"error: invalid source spec {spec!r}. Expected FILE:WEIGHT",
                    file=sys.stderr,
                )
                sys.exit(2)
    else:
        print(
            "error: provide a config file or --input FILE:WEIGHT pairs",
            file=sys.stderr,
        )
        sys.exit(2)

    # CLI flags override config file values
    output = args.output or options.get("output")
    total = args.total if args.total is not None else options.get("total")
    seed = args.seed if args.seed is not None else options.get("seed", 42)
    oversample = args.oversample or options.get("oversample", False)
    sampler = args.sampler or options.get("sampler", "v2")
    by = args.by or options.get("by", "rows")
    tokenizer = args.tokenizer or options.get("tokenizer")
    by_sample = args.by_sample if args.by_sample is not None else options.get("by_sample")
    max_tokens = args.max_tokens if args.max_tokens is not None else options.get("max_tokens")

    if output is None:
        print("error: --output / -o is required (or set 'output' in config)", file=sys.stderr)
        sys.exit(2)
    problem = _option_problem(
        args, by=by, total=total, sampler=sampler, tokenizer=tokenizer,
        by_sample=by_sample, max_tokens=max_tokens,
    )  # fmt: skip
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        sys.exit(2)

    try:
        result = mix_files(
            sources,
            output,
            total=total,
            seed=seed,
            oversample=oversample,
            encoding=args.encoding,
            sampler=sampler,
            by=by,
            tokenizer=tokenizer,
            by_sample=by_sample,
            max_tokens=max_tokens,
        )
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    for line in mix_summary(result, oversample=oversample):
        print(line, file=sys.stderr)

    if not args.no_recipe:
        sidecar = write_mix_recipe(result)
        print(f"recipe:        {sidecar}", file=sys.stderr)
