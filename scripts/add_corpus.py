#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append new corpus files into data/raw/corpus.txt")
    parser.add_argument("inputs", nargs="+", help="New corpus file path(s) to append")
    parser.add_argument(
        "--output",
        default="data/raw/corpus.txt",
        help="Target raw corpus path (default: data/raw/corpus.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    appended_lines = 0
    appended_files = 0

    with output_path.open("a", encoding="utf-8") as out:
        for input_name in args.inputs:
            input_path = Path(input_name).resolve()
            if input_path == output_path:
                print(f"skip_same_file={input_path}")
                continue

            print(f"append_from={input_path}")
            with input_path.open("r", encoding="utf-8", errors="ignore") as src:
                for line in src:
                    out.write(line)
                    appended_lines += 1
            appended_files += 1

    print(f"output={output_path}")
    print(f"appended_files={appended_files}")
    print(f"appended_lines={appended_lines}")


if __name__ == "__main__":
    main()
