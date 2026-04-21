#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from gpt2_pretrain.config import ensure_dirs, load_config
from gpt2_pretrain.data import build_memmap_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode cleaned corpus into train/valid memmap bins.")
    parser.add_argument("--config", required=True, help="Config yaml path")
    parser.add_argument("--input", default=None, help="Override cleaned corpus path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    input_path = args.input or config.data.cleaned_corpus_path
    stats = build_memmap_dataset(
        input_path=input_path,
        tokenizer_path=config.data.tokenizer_path,
        train_bin_path=config.data.train_bin,
        valid_bin_path=config.data.valid_bin,
        train_split=config.data.train_split,
    )
    print(f"train_tokens={stats['train_tokens']}")
    print(f"valid_tokens={stats['valid_tokens']}")


if __name__ == "__main__":
    main()
