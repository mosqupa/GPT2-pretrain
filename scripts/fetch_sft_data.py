#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from datasets import load_dataset


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and normalize Chinese SFT data from Hugging Face.")
    parser.add_argument(
        "--dataset",
        default="PKU-Alignment/Align-Anything-Instruction-100K-zh",
        help="Hugging Face dataset id",
    )
    parser.add_argument("--config", default=None, help="Dataset config name if needed")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--output", default="data/raw/sft_zh.jsonl", help="Normalized jsonl output path")
    parser.add_argument("--max-samples", type=int, default=50000, help="Maximum kept samples, 0 means all")
    parser.add_argument("--hf-endpoint", default=None, help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--hf-timeout", type=int, default=120, help="Hugging Face timeout in seconds")
    parser.add_argument("--hf-retries", type=int, default=10, help="Retry times on transient network failures")
    parser.add_argument("--streaming", dest="streaming", action="store_true", help="Enable streaming mode")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable streaming mode")
    parser.set_defaults(streaming=True)
    return parser.parse_args()


def _conversation_message_text(message: dict) -> tuple[str, str]:
    role = str(message.get("role") or message.get("from") or "").strip().lower()
    content = clean_text(str(message.get("content") or message.get("value") or ""))
    return role, content


def normalize_record(sample: dict) -> dict | None:
    if "instruction" in sample and "output" in sample:
        instruction = clean_text(str(sample.get("instruction") or ""))
        input_text = clean_text(str(sample.get("input") or ""))
        output = clean_text(str(sample.get("output") or ""))
        if instruction and output:
            return {"instruction": instruction, "input": input_text, "output": output}

    if isinstance(sample.get("prompt"), str) and isinstance(sample.get("response"), str):
        instruction = clean_text(sample["prompt"])
        output = clean_text(sample["response"])
        if instruction and output:
            return {"instruction": instruction, "input": "", "output": output}

    conversations = sample.get("conversations")
    if isinstance(conversations, list):
        user_parts = []
        assistant_reply = ""
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role, content = _conversation_message_text(message)
            if not content:
                continue
            if role in {"user", "human"}:
                user_parts.append(content)
            elif role in {"assistant", "gpt"} and not assistant_reply:
                assistant_reply = content
        if user_parts and assistant_reply:
            return {"instruction": "\n".join(user_parts), "input": "", "output": assistant_reply}

    prompt = sample.get("prompt")
    completion = sample.get("completion")
    if isinstance(prompt, list) and isinstance(completion, list):
        prompt_parts = []
        answer_parts = []
        for item in prompt:
            if isinstance(item, dict):
                role, content = _conversation_message_text(item)
                if role in {"user", "human", "system"} and content:
                    prompt_parts.append(content)
        for item in completion:
            if isinstance(item, dict):
                role, content = _conversation_message_text(item)
                if role in {"assistant", "gpt"} and content:
                    answer_parts.append(content)
        if prompt_parts and answer_parts:
            return {"instruction": "\n".join(prompt_parts), "input": "", "output": "\n".join(answer_parts)}

    return None


def load_dataset_with_retry(args: argparse.Namespace):
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(args.hf_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.hf_timeout)

    kwargs = {
        "path": args.dataset,
        "name": args.config,
        "split": args.split,
        "streaming": args.streaming,
    }

    last_exc = None
    for attempt in range(1, args.hf_retries + 1):
        try:
            return load_dataset(**kwargs)
        except Exception as exc:
            last_exc = exc
            message = str(exc)
            transient = (
                "ReadTimeout" in message
                or "Read timed out" in message
                or "ConnectionError" in message
                or "Couldn't reach" in message
                or "LocalEntryNotFoundError" in message
            )
            if not transient or attempt == args.hf_retries:
                raise
            wait_sec = min(5 * attempt, 30)
            print(f"[retry {attempt}/{args.hf_retries}] 连接超时，{wait_sec}s 后重试...")
            time.sleep(wait_sec)
    raise RuntimeError(f"Dataset loading failed unexpectedly: {last_exc}")


def main() -> None:
    args = parse_args()
    dataset = load_dataset_with_retry(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for sample in dataset:
            record = normalize_record(sample)
            if record is None:
                skipped += 1
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 5000 == 0:
                print(f"kept={kept} skipped={skipped}")
            if args.max_samples > 0 and kept >= args.max_samples:
                break

    print(f"dataset={args.dataset}")
    print(f"output={output_path}")
    print(f"kept_samples={kept}")
    print(f"skipped_samples={skipped}")


if __name__ == "__main__":
    main()
