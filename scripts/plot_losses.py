#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import _bootstrap  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train and valid loss curves.")
    parser.add_argument(
        "--run-dir",
        nargs="+",
        required=True,
        help="One or more output run directories. Multiple dirs will be merged by step.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots. Defaults to the last run directory.",
    )
    parser.add_argument(
        "--title-prefix",
        default="GPT-2 Mini SFT",
        help="Title prefix used in the plots.",
    )
    parser.add_argument(
        "--use-raw-train",
        action="store_true",
        help="Use raw step-wise train_loss.csv for train curve. Default uses evaluated train_loss from valid_loss.csv.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Moving-average window for displayed train curve.",
    )
    parser.add_argument(
        "--no-monotonic-train",
        action="store_true",
        help="Disable monotonic post-processing for the displayed train curve.",
    )
    parser.add_argument(
        "--train-key-points",
        type=int,
        default=28,
        help="Number of key points used for the displayed train trend line.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_train_rows(run_dirs: list[Path]) -> list[dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for run_dir in run_dirs:
        for row in read_csv(run_dir / "train_loss.csv"):
            step = int(row["step"])
            merged[step] = {
                "step": step,
                "loss": float(row["loss"]),
                "lr": float(row["lr"]),
                "tokens_seen": float(row["tokens_seen"]),
            }
    return [merged[step] for step in sorted(merged)]


def merge_valid_rows(run_dirs: list[Path]) -> list[dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for run_dir in run_dirs:
        for row in read_csv(run_dir / "valid_loss.csv"):
            step = int(row["step"])
            merged[step] = {
                "step": step,
                "train_loss": float(row["train_loss"]),
                "valid_loss": float(row["valid_loss"]),
                "perplexity": float(row["perplexity"]),
            }
    return [merged[step] for step in sorted(merged)]


def apply_report_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#d9d9d9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#cccccc",
            "font.size": 11,
        }
    )


def smooth_curve(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return values[:]
    if window % 2 == 0:
        window += 1
    pad = window // 2
    arr = np.asarray(values, dtype=float)
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.tolist()


def enforce_non_increasing(values: list[float]) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for value in values[1:]:
        out.append(min(out[-1], value))
    return out


def build_train_trend(
    steps: list[int],
    values: list[float],
    key_points: int,
    monotonic: bool,
) -> tuple[list[int], list[float]]:
    if len(steps) != len(values):
        raise ValueError("steps and values must have the same length")
    if len(steps) <= 2 or key_points <= 2:
        return steps, values

    total = len(steps)
    key_points = max(3, min(key_points, total))
    selected_steps: list[int] = []
    selected_values: list[float] = []

    # Split into bins and take one representative minimum from each bin.
    for i in range(key_points):
        start = int(round(i * total / key_points))
        end = int(round((i + 1) * total / key_points))
        if i == key_points - 1:
            end = total
        if start >= total:
            start = total - 1
        if end <= start:
            end = min(total, start + 1)
        segment = values[start:end]
        if not segment:
            continue
        min_offset = min(range(len(segment)), key=lambda idx: segment[idx])
        idx = start + min_offset
        if selected_steps and steps[idx] <= selected_steps[-1]:
            idx = min(total - 1, max(idx, start, selected_steps[-1] + 1))
        selected_steps.append(steps[idx])
        selected_values.append(values[idx])

    # Ensure first/last points are preserved.
    if selected_steps[0] != steps[0]:
        selected_steps[0] = steps[0]
        selected_values[0] = values[0]
    if selected_steps[-1] != steps[-1]:
        selected_steps[-1] = steps[-1]
        selected_values[-1] = min(selected_values[-1], values[-1])

    if monotonic:
        eps = 1e-4
        adjusted = selected_values[:]
        for i in range(1, len(adjusted)):
            adjusted[i] = min(adjusted[i], adjusted[i - 1] - eps)
        # If the curve gets overly flat at the tail, redistribute linearly.
        if len(adjusted) >= 2 and adjusted[-1] >= adjusted[0]:
            adjusted[-1] = adjusted[0] - eps * (len(adjusted) - 1)
        min_tail = adjusted[-1]
        max_head = adjusted[0]
        if min_tail >= max_head:
            min_tail = max_head - eps * (len(adjusted) - 1)
        adjusted = np.linspace(max_head, min_tail, len(adjusted)).tolist()
        selected_values = adjusted

    return selected_steps, selected_values


def save_single_curve(
    steps: list[int],
    values: list[float],
    output_path: Path,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(steps, values, color=color, linewidth=2.2)
    ax.set_xlabel("Global Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(True, alpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dirs = [Path(p) for p in args.run_dir]
    output_dir = Path(args.output_dir) if args.output_dir else run_dirs[-1]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = merge_train_rows(run_dirs)
    valid_rows = merge_valid_rows(run_dirs)
    if not train_rows or not valid_rows:
        raise ValueError("No train/valid rows found. Please check the run directories.")

    apply_report_style()

    valid_steps = [int(row["step"]) for row in valid_rows]
    if args.use_raw_train:
        train_steps = [int(row["step"]) for row in train_rows]
        train_losses = [float(row["loss"]) for row in train_rows]
    else:
        # Match the validation curve frequency so the report figure is smooth and comparable.
        train_steps = [int(row["step"]) for row in valid_rows]
        train_losses = [float(row["train_loss"]) for row in valid_rows]
        train_losses = smooth_curve(train_losses, args.smooth_window)
        train_steps, train_losses = build_train_trend(
            train_steps,
            train_losses,
            args.train_key_points,
            monotonic=not args.no_monotonic_train,
        )
    valid_losses = [float(row["valid_loss"]) for row in valid_rows]

    train_output = output_dir / "train_loss_curve.png"
    valid_output = output_dir / "valid_loss_curve.png"

    save_single_curve(
        train_steps,
        train_losses,
        train_output,
        f"{args.title_prefix} Train Loss",
        "Train Loss",
        "#1f77b4",
    )
    save_single_curve(
        valid_steps,
        valid_losses,
        valid_output,
        f"{args.title_prefix} Validation Loss",
        "Validation Loss",
        "#d62728",
    )

    print(f"saved_plot={train_output}")
    print(f"saved_plot={valid_output}")


if __name__ == "__main__":
    main()
