#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  # pyright: ignore[reportUnusedImport]
from gpt2_pretrain.data import clean_corpus_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw corpus file line by line.")
    parser.add_argument("--input", required=True, help="Path to raw text corpus")
    parser.add_argument("--output", required=True, help="Path to cleaned corpus")
    parser.add_argument("--min-length", type=int, default=6, help="Minimum kept line length")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable exact-line dedup for huge corpora")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kept = clean_corpus_file(args.input, args.output, min_length=args.min_length, dedupe=not args.no_dedupe)
    print(f"cleaned_corpus={args.output}")
    print(f"kept_lines={kept}")


if __name__ == "__main__":
    main()
