#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from gpt2_pretrain.config import ensure_dirs, load_config
from gpt2_pretrain.tokenizer_utils import train_bpe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for Chinese corpus.")
    parser.add_argument("--config", required=True, help="Config yaml path")
    parser.add_argument("--input", default=None, help="Override input corpus path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    input_path = args.input or config.data.cleaned_corpus_path
    tokenizer = train_bpe_tokenizer(
        input_path=input_path,
        output_path=config.data.tokenizer_path,
        vocab_size=config.data.vocab_size,
        min_frequency=config.data.min_frequency,
    )
    print(f"tokenizer_path={config.data.tokenizer_path}")
    print(f"vocab_size={tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
