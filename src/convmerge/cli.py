"""CLI entrypoint for convmerge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from convmerge import __version__
from convmerge.convert import convert_file


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="convmerge",
        description="Merge heterogeneous chat/text sources into a single LLM training format.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command", required=True)

    convert_p = subparsers.add_parser(
        "convert",
        help="Convert a JSONL file using a source adapter and output format",
    )
    convert_p.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input JSONL path (one JSON object per line)",
    )
    convert_p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSONL path",
    )
    convert_p.add_argument(
        "--from",
        dest="adapter",
        required=True,
        metavar="ADAPTER",
        help="Source adapter: alpaca, sharegpt, ...",
    )
    convert_p.add_argument(
        "--format",
        "-f",
        dest="output_format",
        required=True,
        metavar="FORMAT",
        help="Output format: messages, alpaca, ...",
    )
    convert_p.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8)",
    )

    args = parser.parse_args()

    if args.command == "convert":
        if not args.input.is_file():
            print(f"error: input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        n_in, n_out = convert_file(
            args.input,
            args.output,
            adapter_name=args.adapter,
            output_format=args.output_format,
            encoding=args.encoding,
        )
        print(f"read {n_in} lines, wrote {n_out} examples", file=sys.stderr)


if __name__ == "__main__":
    main()
