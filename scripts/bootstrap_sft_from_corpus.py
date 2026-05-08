#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap offline Chinese SFT data from local corpus.")
    parser.add_argument("--input", default="data/raw/corpus.txt", help="Local corpus path, one sample per line")
    parser.add_argument("--output", default="data/raw/sft_zh.jsonl", help="Output SFT jsonl path")
    parser.add_argument("--max-samples", type=int, default=50000, help="Maximum SFT samples to generate")
    parser.add_argument("--min-length", type=int, default=30, help="Minimum source line length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


TOPIC_SPLIT_PATTERNS = [
    "是",
    "位于",
    "为",
    "指",
    "属于",
    "又称",
    "简称",
    "是一",
]


def extract_topic(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""

    prefix = text
    for marker in TOPIC_SPLIT_PATTERNS:
        idx = prefix.find(marker)
        if idx > 0:
            prefix = prefix[:idx]
            break

    prefix = prefix.split("，")[0]
    prefix = prefix.split(",")[0]
    prefix = prefix.split("。")[0]
    prefix = prefix.strip("：:;；,，.。[]【】()（）\"'“”‘’ ")
    if 1 <= len(prefix) <= 40:
        return prefix
    return ""


def build_instruction(topic: str, rng: random.Random) -> str:
    templates = [
        "请简要介绍一下{topic}。",
        "什么是{topic}？请用中文简明说明。",
        "请概括说明{topic}的基本情况。",
        "请写一段关于{topic}的简介。",
        "请用通俗的话介绍{topic}。",
    ]
    return rng.choice(templates).format(topic=topic)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8", errors="ignore") as src, output_path.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            line = clean_text(raw_line)
            if len(line) < args.min_length:
                skipped += 1
                continue
            topic = extract_topic(line)
            if not topic:
                skipped += 1
                continue

            record = {
                "instruction": build_instruction(topic, rng),
                "input": "",
                "output": line,
            }
            dedupe_key = (record["instruction"], record["output"])
            if dedupe_key in seen:
                skipped += 1
                continue
            seen.add(dedupe_key)
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 5000 == 0:
                print(f"kept={kept} skipped={skipped}")
            if args.max_samples > 0 and kept >= args.max_samples:
                break

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"kept_samples={kept}")
    print(f"skipped_samples={skipped}")


if __name__ == "__main__":
    main()
