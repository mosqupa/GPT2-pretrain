#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from gpt2_pretrain.config import load_config
from gpt2_pretrain.tokenizer_utils import load_tokenizer
from gpt2_pretrain.trainer import build_model
from gpt2_pretrain.utils import detect_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--config", required=True, help="Config yaml path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--prompt", default="人工智能", help="Prompt text")
    parser.add_argument(
        "--strategy",
        choices=["greedy", "top_k", "top_p"],
        default="top_p",
        help="Decoding strategy",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = detect_device(config.train.device)
    tokenizer = load_tokenizer(config.data.tokenizer_path)
    model = build_model(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    encoded = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    max_new_tokens = args.max_new_tokens or config.generation.max_new_tokens
    temperature = args.temperature if args.temperature is not None else config.generation.temperature

    top_k = None
    top_p = None
    if args.strategy == "greedy":
        temperature = 0.0
    elif args.strategy == "top_k":
        top_k = config.generation.top_k
    elif args.strategy == "top_p":
        top_p = config.generation.top_p

    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    print(text)

    output_path = Path(config.paths.sample_dir) / f"generate_{args.strategy}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
