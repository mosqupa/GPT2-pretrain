#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train and valid loss curves.")
    parser.add_argument("--run-dir", required=True, help="Output run directory")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    train_rows = read_csv(run_dir / "train_loss.csv")
    valid_rows = read_csv(run_dir / "valid_loss.csv")

    train_steps = [int(row["step"]) for row in train_rows]
    train_losses = [float(row["loss"]) for row in train_rows]
    valid_steps = [int(row["step"]) for row in valid_rows]
    valid_losses = [float(row["valid_loss"]) for row in valid_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_losses, label="train_loss")
    plt.plot(valid_steps, valid_losses, label="valid_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    output_path = run_dir / "loss_curve.png"
    plt.savefig(output_path, dpi=200)
    print(f"saved_plot={output_path}")


if __name__ == "__main__":
    main()
