#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple configs sequentially for comparison.")
    parser.add_argument("--configs", nargs="+", required=True, help="List of config yaml paths")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for config_path in args.configs:
        completed = subprocess.run(
            ["python3", "scripts/train.py", "--config", config_path],
            check=True,
            capture_output=True,
            text=True,
        )
        print(completed.stdout)
        metrics = json.loads(completed.stdout)
        results.append(metrics)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
