from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gpt2_pretrain.tokenizer_utils import load_tokenizer


SFT_IGNORE_INDEX = -100


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


def _read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_sft_prompt(instruction: str, input_text: str = "", system_prompt: str = "") -> str:
    sections = []
    system_prompt = clean_text(system_prompt)
    if system_prompt:
        sections.append(f"### 系统\n{system_prompt}")
    sections.append(f"### 指令\n{clean_text(instruction)}")
    input_text = clean_text(input_text)
    if input_text:
        sections.append(f"### 输入\n{input_text}")
    sections.append("### 回答\n")
    return "\n\n".join(sections)


def split_sft_jsonl(
    input_path: str | Path,
    train_path: str | Path,
    valid_path: str | Path,
    valid_split: float,
    seed: int,
    max_samples: int = 0,
) -> dict[str, int]:
    if not 0 < valid_split < 1:
        raise ValueError(f"valid_split must be between 0 and 1, got {valid_split}")

    seen = set()
    records: list[dict] = []
    for record in _read_jsonl(input_path):
        instruction = clean_text(str(record.get("instruction", "")))
        input_text = clean_text(str(record.get("input", "")))
        output_text = clean_text(str(record.get("output", "")))
        if not instruction or not output_text:
            continue
        dedupe_key = (instruction, input_text, output_text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append({"instruction": instruction, "input": input_text, "output": output_text})

    rng = random.Random(seed)
    rng.shuffle(records)
    if max_samples > 0:
        records = records[:max_samples]

    valid_count = max(1, int(len(records) * valid_split))
    if valid_count >= len(records):
        valid_count = max(1, len(records) - 1)
    train_records = records[valid_count:]
    valid_records = records[:valid_count]
    if not train_records or not valid_records:
        raise ValueError("SFT split produced empty train or valid set.")

    write_jsonl(train_path, train_records)
    write_jsonl(valid_path, valid_records)
    return {"train_samples": len(train_records), "valid_samples": len(valid_records)}


class SFTJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, system_prompt: str = "", max_samples: int = 0) -> None:
        self.samples: list[dict[str, str]] = []
        for idx, record in enumerate(_read_jsonl(jsonl_path)):
            instruction = clean_text(str(record.get("instruction", "")))
            input_text = clean_text(str(record.get("input", "")))
            output_text = clean_text(str(record.get("output", "")))
            if not instruction or not output_text:
                continue
            self.samples.append(
                {
                    "prompt": format_sft_prompt(instruction, input_text, system_prompt),
                    "response": output_text,
                }
            )
            if max_samples > 0 and len(self.samples) >= max_samples:
                break
        if not self.samples:
            raise ValueError(f"No valid SFT samples found in {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.samples[idx]


class SFTCollator:
    def __init__(self, tokenizer_path: str | Path, seq_length: int, prompt_loss: bool = False) -> None:
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.seq_length = seq_length
        self.prompt_loss = prompt_loss
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.bos_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        if self.pad_id is None or self.bos_id is None or self.eos_id is None:
            raise ValueError("Tokenizer is missing required special tokens for SFT.")

    def _encode_plain(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def __call__(self, batch: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        input_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        max_len = 0

        for sample in batch:
            prompt_ids = self._encode_plain(sample["prompt"])
            response_ids = self._encode_plain(sample["response"])
            max_content_len = max(1, self.seq_length - 2)
            if len(prompt_ids) + len(response_ids) > max_content_len:
                max_prompt_len = min(len(prompt_ids), max_content_len - 1)
                prompt_ids = prompt_ids[:max_prompt_len]
                response_ids = response_ids[: max_content_len - len(prompt_ids)]
            if not response_ids:
                continue

            full_ids = [self.bos_id] + prompt_ids + response_ids + [self.eos_id]
            label_ids = full_ids.copy() if self.prompt_loss else [SFT_IGNORE_INDEX] * (1 + len(prompt_ids)) + response_ids + [self.eos_id]
            x_ids = full_ids[:-1]
            y_ids = label_ids[1:]
            if not any(token != SFT_IGNORE_INDEX for token in y_ids):
                continue

            x = torch.tensor(x_ids, dtype=torch.long)
            y = torch.tensor(y_ids, dtype=torch.long)
            input_rows.append(x)
            label_rows.append(y)
            max_len = max(max_len, x.size(0))

        if not input_rows:
            raise ValueError("SFT batch is empty after truncation/filtering.")

        batch_x = torch.full((len(input_rows), max_len), self.pad_id, dtype=torch.long)
        batch_y = torch.full((len(label_rows), max_len), SFT_IGNORE_INDEX, dtype=torch.long)
        for idx, (x, y) in enumerate(zip(input_rows, label_rows)):
            batch_x[idx, : x.size(0)] = x
            batch_y[idx, : y.size(0)] = y
        return batch_x, batch_y


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
