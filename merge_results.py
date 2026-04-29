#!/usr/bin/env python3
"""Merge two-phase training results into single folders.

Each experiment was trained in two phases:
  - Base run: epochs 0-36 (37 epochs)
  - Extended run: continued from epoch 36 checkpoint, logged as epochs 36-50

The extended run's epoch counter starts at the checkpoint epoch, so its
"Epoch 36" is one training step after the base's "Epoch 36". All extended
epochs are shifted by +1 to produce contiguous numbering (0-51, 52 epochs).
"""

import math
import os
import re
import shutil

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

PAIRS = [
    ("experiment1_baseline", "experiment1_baseline_50epoch", "experiment1_baseline_merged"),
    ("experiment1_baseline_with_crop_37epoch", "experiment1_baseline_with_crop_50epoch", "experiment1_baseline_with_crop_merged"),
    ("experiment2_crop_diffusion", "experiment2_crop_diffusion_50epoch", "experiment2_crop_diffusion_merged"),
    ("experiment3_SelfAttn", "experiment3_SelfAttn_50epoch", "experiment3_SelfAttn_merged"),
]


def parse_train_log(file_path):
    """Parse a train/val log file. Returns (epochs, metrics_dict)."""
    epochs = []
    metrics = {}

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        if line.startswith("Epoch"):
            parts = line.split()
            if len(parts) >= 2:
                epochs.append(int(parts[1]))
            continue

        if "=" not in line:
            continue

        for pair in line.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                continue
            try:
                num_value = float(value)
            except ValueError:
                continue
            metrics.setdefault(key, []).append(num_value)

    return epochs, metrics


def plot_metrics(epochs, metrics, output_path):
    """Plot all metrics into a single figure and save to output_path."""
    valid_items = [(k, v) for k, v in metrics.items() if len(v) == len(epochs)]
    if not valid_items:
        print(f"  Warning: no valid metrics for {output_path}")
        return

    n = len(valid_items)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.8 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (metric_name, values) in enumerate(valid_items):
        ax = axes_flat[idx]
        ax.plot(epochs, values, marker="o", markersize=2.5, linewidth=1.3)
        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.3)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Training Metrics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def read_raw_entries(filepath):
    """Read log file into list of (epoch_num, epoch_line, metric_line)."""
    with open(filepath) as f:
        lines = f.readlines()

    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line.startswith("Epoch"):
            parts = line.split()
            epoch_num = int(parts[1])
            metric_line = lines[i + 1].rstrip("\n") if i + 1 < len(lines) else ""
            entries.append((epoch_num, line, metric_line))
            i += 2
        else:
            i += 1
    return entries


def merge_logs(base_path, extended_path, output_path):
    """Merge a log file from base and extended folders.

    Base epochs kept as-is. Extended epochs shifted by +1 so that
    extended "Epoch 36" becomes 37, etc.
    """
    base_entries = read_raw_entries(base_path)
    ext_entries = read_raw_entries(extended_path)

    merged = list(base_entries)
    for epoch_num, epoch_line, metric_line in ext_entries:
        new_num = epoch_num + 1
        new_line = re.sub(r"Epoch \d+ :", f"Epoch {new_num} :", epoch_line)
        merged.append((new_num, new_line, metric_line))

    with open(output_path, "w") as f:
        f.write("-----------------------Date: merged---------------------\n")
        for _, epoch_line, metric_line in merged:
            f.write(f"{epoch_line}\n")
            f.write(f"{metric_line}\n")

    return len(merged)


def merge_pair(base_name, ext_name, out_name):
    """Merge a single experiment pair."""
    base = os.path.join(RESULTS_DIR, base_name)
    ext = os.path.join(RESULTS_DIR, ext_name)
    out = os.path.join(RESULTS_DIR, out_name)

    os.makedirs(out, exist_ok=True)

    # 1. Merge train_log.txt
    n_train = merge_logs(
        os.path.join(base, "train_log.txt"),
        os.path.join(ext, "train_log.txt"),
        os.path.join(out, "train_log.txt"),
    )

    # 2. Merge val_log.txt
    n_val = merge_logs(
        os.path.join(base, "val_log.txt"),
        os.path.join(ext, "val_log.txt"),
        os.path.join(out, "val_log.txt"),
    )

    # 3. Regenerate metrics plots from merged logs
    for log_name, png_name, title in [
        ("train_log.txt", "train_metrics.png", "Training Metrics"),
        ("val_log.txt", "val_metrics.png", "Validation Metrics"),
    ]:
        log_path = os.path.join(out, log_name)
        out_path = os.path.join(out, png_name)
        epochs, metrics = parse_train_log(log_path)
        plot_metrics(epochs, metrics, out_path)

    # 4. Copy test results (prefer extended = latest model)
    copied = []
    for fname in sorted(os.listdir(ext)):
        if fname.startswith("test_") and (
            fname.endswith("_log.txt") or fname.endswith("_params.json")
        ):
            shutil.copy2(os.path.join(ext, fname), os.path.join(out, fname))
            copied.append(fname)

    # Fallback: copy from base if extended had none
    if not copied:
        for fname in sorted(os.listdir(base)):
            if fname.startswith("test_") and (
                fname.endswith("_log.txt") or fname.endswith("_params.json")
            ):
                shutil.copy2(os.path.join(base, fname), os.path.join(out, fname))
                copied.append(fname)

    # 5. Write merged train_params.txt (extended as base, update counts)
    with open(os.path.join(ext, "train_params.txt")) as f:
        params_lines = f.readlines()

    with open(os.path.join(out, "train_params.txt"), "w") as f:
        for line in params_lines:
            if line.startswith("epoch_number"):
                f.write(f"epoch_number = {n_train}\n")
            elif line.startswith("train_continue ="):
                f.write("train_continue = False\n")
            elif line.startswith("train_continue_path") or line.startswith("train_continue_epoch"):
                continue  # drop continuation-specific fields
            else:
                f.write(line)

    print(f"  {out_name}: {n_train} train epochs, {n_val} val epochs, {len(copied)} test files")


def main():
    for base_name, ext_name, out_name in PAIRS:
        merge_pair(base_name, ext_name, out_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
