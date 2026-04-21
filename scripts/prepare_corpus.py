#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from gpt2_pretrain.data import clean_corpus_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw corpus file line by line.")
    parser.add_argument("--input", required=True, help="Path to raw text corpus")
    parser.add_argument("--output", required=True, help="Path to cleaned corpus")
    parser.add_argument("--min-length", type=int, default=6, help="Minimum kept line length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kept = clean_corpus_file(args.input, args.output, min_length=args.min_length)
    print(f"cleaned_corpus={args.output}")
    print(f"kept_lines={kept}")


if __name__ == "__main__":
    main()
