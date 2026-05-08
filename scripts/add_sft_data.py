#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def normalize_record(sample: dict) -> dict | None:
    instruction = clean_text(sample.get("instruction", ""))
    input_text = clean_text(sample.get("input", ""))
    output = clean_text(sample.get("output", ""))
    if instruction and output:
        return {"instruction": instruction, "input": input_text, "output": output}

    if isinstance(sample.get("prompt"), str) and isinstance(sample.get("response"), str):
        instruction = clean_text(sample.get("prompt", ""))
        output = clean_text(sample.get("response", ""))
        if instruction and output:
            return {"instruction": instruction, "input": "", "output": output}

    conversations = sample.get("conversations")
    if isinstance(conversations, list):
        user_parts = []
        assistant_reply = None
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("from") or "").strip().lower()
            content = clean_text(message.get("content") or message.get("value") or "")
            if not content:
                continue
            if role in {"user", "human"}:
                user_parts.append(content)
            elif role in {"assistant", "gpt"} and assistant_reply is None:
                assistant_reply = content
        if user_parts and assistant_reply:
            return {"instruction": "\n".join(user_parts), "input": "", "output": assistant_reply}

    return None


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"skip_invalid_json={path}:{line_no}")
                continue
            if not isinstance(payload, dict):
                print(f"skip_non_object={path}:{line_no}")
                continue
            yield payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge uploaded SFT jsonl files into data/raw/sft_zh.jsonl")
    parser.add_argument("inputs", nargs="+", help="New SFT jsonl file path(s) to merge")
    parser.add_argument(
        "--output",
        default="data/raw/sft_zh.jsonl",
        help="Target raw SFT jsonl path (default: data/raw/sft_zh.jsonl)",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable exact deduplication on instruction/input/output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dedupe = not args.no_dedupe
    seen = set()
    existing_records = 0

    if output_path.exists():
        for record in iter_jsonl(output_path):
            normalized = normalize_record(record)
            if normalized is None:
                continue
            if dedupe:
                seen.add((normalized["instruction"], normalized["input"], normalized["output"]))
            existing_records += 1

    appended_files = 0
    appended_records = 0
    skipped_duplicates = 0
    skipped_invalid = 0

    with output_path.open("a", encoding="utf-8") as out:
        for input_name in args.inputs:
            input_path = Path(input_name).resolve()
            if input_path == output_path:
                print(f"skip_same_file={input_path}")
                continue

            print(f"merge_from={input_path}")
            file_appended = 0
            for record in iter_jsonl(input_path):
                normalized = normalize_record(record)
                if normalized is None:
                    skipped_invalid += 1
                    continue

                key = (normalized["instruction"], normalized["input"], normalized["output"])
                if dedupe and key in seen:
                    skipped_duplicates += 1
                    continue

                if dedupe:
                    seen.add(key)
                out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                appended_records += 1
                file_appended += 1

            appended_files += 1
            print(f"appended_from_file={file_appended}")

    print(f"output={output_path}")
    print(f"existing_records={existing_records}")
    print(f"appended_files={appended_files}")
    print(f"appended_records={appended_records}")
    print(f"skipped_duplicates={skipped_duplicates}")
    print(f"skipped_invalid={skipped_invalid}")
    print(f"total_records={existing_records + appended_records}")


if __name__ == "__main__":
    main()
