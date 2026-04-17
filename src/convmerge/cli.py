"""CLI entrypoint for convmerge."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="convmerge",
        description="Merge heterogeneous chat/text sources into a single LLM training format.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.parse_args()
    print("convmerge: no commands yet — package skeleton only.")


if __name__ == "__main__":
    main()
