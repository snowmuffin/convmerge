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
        "--no-recipe",
        action="store_true",
        help="Skip writing the .mix.json sidecar file",
    )
    p.add_argument("--encoding", default="utf-8")


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

    if output is None:
        print("error: --output / -o is required (or set 'output' in config)", file=sys.stderr)
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
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    for s in result.sources:
        clipped = s.available < s.requested and not oversample
        note = f" (clipped from {s.requested:,})" if clipped else ""
        print(
            f"  {s.path}: weight={s.weight:.4f} written={s.written:,}{note}",
            file=sys.stderr,
        )
    print(f"total written: {result.total_written:,} -> {result.output}", file=sys.stderr)

    if not args.no_recipe:
        sidecar = write_mix_recipe(result, encoding=args.encoding)
        print(f"recipe:        {sidecar}", file=sys.stderr)
