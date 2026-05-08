#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from gpt2_pretrain.config import ensure_dirs, load_config
from gpt2_pretrain.data import split_sft_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split normalized SFT jsonl into train/valid sets.")
    parser.add_argument("--config", required=True, help="Config yaml path")
    parser.add_argument("--input", default=None, help="Override source SFT jsonl path")
    parser.add_argument("--train-output", default=None, help="Override SFT train jsonl path")
    parser.add_argument("--valid-output", default=None, help="Override SFT valid jsonl path")
    parser.add_argument("--max-samples", type=int, default=None, help="Override config.sft.max_samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    if config.sft is None:
        raise ValueError("Config is missing `sft` section.")

    input_path = args.input or config.sft.source_file
    train_path = args.train_output or config.sft.train_file
    valid_path = args.valid_output or config.sft.valid_file
    if not input_path or not train_path or not valid_path:
        raise ValueError("SFT source/train/valid paths must be provided.")

    stats = split_sft_jsonl(
        input_path=input_path,
        train_path=train_path,
        valid_path=valid_path,
        valid_split=config.sft.valid_split,
        seed=config.seed,
        max_samples=args.max_samples if args.max_samples is not None else config.sft.max_samples,
    )
    print(f"sft_train={train_path}")
    print(f"sft_valid={valid_path}")
    print(f"train_samples={stats['train_samples']}")
    print(f"valid_samples={stats['valid_samples']}")


if __name__ == "__main__":
    main()
