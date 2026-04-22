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


def clean_corpus_file(
    input_path: str | Path,
    output_path: str | Path,
    min_length: int = 6,
    dedupe: bool = True,
) -> int:
    seen = set() if dedupe else None
    kept = 0
    # Some downloaded corpora may contain occasional invalid bytes.
    # Ignore undecodable bytes to keep the preprocessing pipeline running.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_path).open("r", encoding="utf-8", errors="ignore") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw_line in src:
            line = clean_text(raw_line)
            if len(line) < min_length:
                continue
            if dedupe:
                assert seen is not None
                if line in seen:
                    continue
                seen.add(line)
            dst.write(line + "\n")
            kept += 1
    return kept


def build_memmap_dataset(
    input_path: str | Path,
    tokenizer_path: str | Path,
    train_bin_path: str | Path,
    valid_bin_path: str | Path,
    train_split: float,
) -> dict[str, int]:
    tokenizer = load_tokenizer(tokenizer_path)
    train_bin_path = Path(train_bin_path)
    valid_bin_path = Path(valid_bin_path)
    train_bin_path.parent.mkdir(parents=True, exist_ok=True)
    valid_bin_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = train_bin_path.parent / "_all_tokens.uint16.bin"

    total_tokens = 0
    line_count = 0
    with Path(input_path).open("r", encoding="utf-8", errors="ignore") as src, staging_path.open("wb") as tmp:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line).ids
            if not ids:
                continue
            np.asarray(ids, dtype=np.uint16).tofile(tmp)
            total_tokens += len(ids)
            line_count += 1
            if line_count % 50_000 == 0:
                print(f"encoded_lines={line_count} total_tokens={total_tokens}")

    if total_tokens < 2:
        raise ValueError("Encoded token count is too small to build train/valid bins.")

    split_idx = int(total_tokens * train_split)
    if split_idx <= 0 or split_idx >= total_tokens:
        raise ValueError(f"Invalid train_split={train_split} for total_tokens={total_tokens}")

    all_tokens = np.memmap(staging_path, dtype=np.uint16, mode="r", shape=(total_tokens,))
    train_memmap = np.memmap(train_bin_path, dtype=np.uint16, mode="w+", shape=(split_idx,))
    valid_memmap = np.memmap(valid_bin_path, dtype=np.uint16, mode="w+", shape=(total_tokens - split_idx,))
    train_memmap[:] = all_tokens[:split_idx]
    valid_memmap[:] = all_tokens[split_idx:]
    train_memmap.flush()
    valid_memmap.flush()
    del all_tokens
    del train_memmap
    del valid_memmap
    staging_path.unlink(missing_ok=True)

    return {"train_tokens": int(split_idx), "valid_tokens": int(total_tokens - split_idx)}


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
