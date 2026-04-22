#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import time

from datasets import load_dataset


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Chinese corpus from Hugging Face datasets.")
    parser.add_argument(
        "--dataset",
        default="0xDing/wikipedia-cn-20230720-filtered",
        help="Hugging Face dataset id, e.g. 0xDing/wikipedia-cn-20230720-filtered",
    )
    parser.add_argument("--config", default=None, help="Dataset config name if needed")
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--hf-timeout",
        type=int,
        default=60,
        help="Hugging Face request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--hf-retries",
        type=int,
        default=6,
        help="Retry times when network timeout/connection issues happen (default: 6)",
    )
    parser.add_argument(
        "--data-files",
        default=None,
        help="Optional data file(s) for datasets that expose multiple files",
    )
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument(
        "--text-column",
        default="completion",
        help="Text field name. Common options: completion / text / content",
    )
    parser.add_argument(
        "--output",
        default="data/raw/corpus.txt",
        help="Output corpus path, one sample per line",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwrite",
    )
    parser.add_argument("--min-length", type=int, default=20, help="Minimum length to keep")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=1_500_000,
        help="Maximum kept lines. Tune by disk budget.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Maximum kept characters. 0 means no limit.",
    )
    parser.add_argument("--streaming", dest="streaming", action="store_true", help="Enable streaming mode")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable streaming mode")
    parser.set_defaults(streaming=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    # Increase Hub API timeout to tolerate slow network/mirror.
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(args.hf_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.hf_timeout)

    kwargs = {
        "path": args.dataset,
        "name": args.config,
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.data_files:
        kwargs["data_files"] = args.data_files
    ds = None
    last_exc = None
    for attempt in range(1, args.hf_retries + 1):
        try:
            ds = load_dataset(**kwargs)
            break
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
                print("ERROR: 无法连接到 Hugging Face Hub。")
                print("建议尝试：")
                print("1) 使用镜像参数：--hf-endpoint https://hf-mirror.com")
                print("2) 或先设置环境变量：export HF_ENDPOINT=https://hf-mirror.com")
                print("3) 或检查代理：export HTTPS_PROXY=http://<host>:<port>")
                print("4) 或改为本地文件模式（先手动下载后再处理）")
                raise
            wait_sec = min(5 * attempt, 30)
            print(f"[retry {attempt}/{args.hf_retries}] 连接超时，{wait_sec}s 后重试...")
            time.sleep(wait_sec)
    if ds is None:
        raise RuntimeError(f"Dataset loading failed unexpectedly: {last_exc}")

    kept = 0
    dropped = 0
    total_chars = 0

    open_mode = "a" if args.append else "w"
    with output_path.open(open_mode, encoding="utf-8") as f:
        for sample in ds:
            raw = sample.get(args.text_column)
            if not isinstance(raw, str):
                dropped += 1
                continue

            text = normalize_text(raw)
            if len(text) < args.min_length:
                dropped += 1
                continue

            f.write(text + "\n")
            kept += 1
            total_chars += len(text)

            if kept % 10_000 == 0:
                print(f"kept={kept} dropped={dropped} total_chars={total_chars}")

            if kept >= args.max_lines:
                break
            if args.max_chars > 0 and total_chars >= args.max_chars:
                break

    print(f"dataset={args.dataset}")
    print(f"text_column={args.text_column}")
    print(f"output={output_path}")
    print(f"append={args.append}")
    print(f"kept_lines={kept}")
    print(f"dropped_lines={dropped}")
    print(f"kept_chars={total_chars}")


if __name__ == "__main__":
    main()
