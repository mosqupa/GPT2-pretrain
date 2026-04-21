from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gpt2_pretrain.tokenizer_utils import load_tokenizer


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return text


def clean_corpus_file(input_path: str | Path, output_path: str | Path, min_length: int = 6) -> int:
    kept_lines = []
    seen = set()
    with Path(input_path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = clean_text(raw_line)
            if len(line) < min_length:
                continue
            if line in seen:
                continue
            seen.add(line)
            kept_lines.append(line)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
    return len(kept_lines)


def build_memmap_dataset(
    input_path: str | Path,
    tokenizer_path: str | Path,
    train_bin_path: str | Path,
    valid_bin_path: str | Path,
    train_split: float,
) -> dict[str, int]:
    tokenizer = load_tokenizer(tokenizer_path)
    text = Path(input_path).read_text(encoding="utf-8")
    encoding = tokenizer.encode(text)
    token_ids = np.array(encoding.ids, dtype=np.uint16)
    split_idx = int(len(token_ids) * train_split)
    train_ids = token_ids[:split_idx]
    valid_ids = token_ids[split_idx:]

    train_bin_path = Path(train_bin_path)
    valid_bin_path = Path(valid_bin_path)
    train_bin_path.parent.mkdir(parents=True, exist_ok=True)
    valid_bin_path.parent.mkdir(parents=True, exist_ok=True)

    train_memmap = np.memmap(train_bin_path, dtype=np.uint16, mode="w+", shape=(len(train_ids),))
    valid_memmap = np.memmap(valid_bin_path, dtype=np.uint16, mode="w+", shape=(len(valid_ids),))
    train_memmap[:] = train_ids
    valid_memmap[:] = valid_ids
    train_memmap.flush()
    valid_memmap.flush()

    return {"train_tokens": int(len(train_ids)), "valid_tokens": int(len(valid_ids))}


class MemmapDataset(Dataset):
    def __init__(self, bin_path: str | Path, seq_length: int) -> None:
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_length = seq_length
        if len(self.data) <= seq_length:
            raise ValueError(f"Dataset at {bin_path} is too short for seq_length={seq_length}")

    def __len__(self) -> int:
        return len(self.data) - self.seq_length - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_length + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y

