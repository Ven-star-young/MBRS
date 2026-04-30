#!/usr/bin/env python3
"""Extend merged crop-experiment logs with 100-epoch continuation data."""

import math
import os
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

# Each: (merged_folder, new_100epoch_folder, epoch_shift)
# _100epoch logs epochs 50..100, shift by +2 so "50" -> merged 52
EXTEND = [
    ("experiment1_baseline_with_crop_merged", "experiment1_baseline_with_crop_100epoch", 2),
    ("experiment2_crop_diffusion_merged", "experiment2_crop_diffusion_100epoch", 2),
]


def read_raw_entries(filepath):
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


def parse_train_log(file_path):
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
            key, value = key.strip(), value.strip()
            if not key or not value:
                continue
            try:
                metrics.setdefault(key, []).append(float(value))
            except ValueError:
                continue
    return epochs, metrics


def plot_metrics(epochs, metrics, output_path):
    valid = [(k, v) for k, v in metrics.items() if len(v) == len(epochs)]
    if not valid:
        print(f"  Warning: no valid metrics for {output_path}")
        return
    n = len(valid)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.8 * rows), squeeze=False)
    for idx, (name, vals) in enumerate(valid):
        ax = axes.flatten()[idx]
        ax.plot(epochs, vals, marker="o", markersize=2.5, linewidth=1.3)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    for idx in range(n, len(axes.flatten())):
        axes.flatten()[idx].axis("off")
    fig.suptitle("Training Metrics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def extend_log(existing_path, new_path, shift):
    """Append new entries (with epoch shift) to existing log."""
    existing = read_raw_entries(existing_path)
    new_entries = read_raw_entries(new_path)

    merged = list(existing)
    for epoch_num, epoch_line, metric_line in new_entries:
        new_num = epoch_num + shift
        new_line = re.sub(r"Epoch \d+ :", f"Epoch {new_num} :", epoch_line)
        merged.append((new_num, new_line, metric_line))

    with open(existing_path, "w") as f:
        f.write("-----------------------Date: merged---------------------\n")
        for _, epoch_line, metric_line in merged:
            f.write(f"{epoch_line}\n")
            f.write(f"{metric_line}\n")

    return len(merged)


def main():
    for merged_name, new_name, shift in EXTEND:
        merged = os.path.join(RESULTS_DIR, merged_name)
        new_dir = os.path.join(RESULTS_DIR, new_name)

        print(f"\nExtending: {merged_name} + {new_name}")

        # 1. Extend train_log.txt
        n_train = extend_log(
            os.path.join(merged, "train_log.txt"),
            os.path.join(new_dir, "train_log.txt"),
            shift,
        )

        # 2. Extend val_log.txt
        n_val = extend_log(
            os.path.join(merged, "val_log.txt"),
            os.path.join(new_dir, "val_log.txt"),
            shift,
        )

        # 3. Regenerate plots
        for log_name, png_name in [("train_log.txt", "train_metrics.png"), ("val_log.txt", "val_metrics.png")]:
            log_path = os.path.join(merged, log_name)
            out_path = os.path.join(merged, png_name)
            epochs, metrics = parse_train_log(log_path)
            plot_metrics(epochs, metrics, out_path)

        # 4. Replace test results with new ones (latest model = best converged)
        for fname in sorted(os.listdir(new_dir)):
            if fname.startswith("test_") and (fname.endswith("_log.txt") or fname.endswith("_params.json")):
                shutil.copy2(os.path.join(new_dir, fname), os.path.join(merged, fname))

        # 5. Update train_params.txt
        params_path = os.path.join(new_dir, "train_params.txt")
        with open(params_path) as f:
            params_lines = f.readlines()
        with open(os.path.join(merged, "train_params.txt"), "w") as f:
            for line in params_lines:
                if line.startswith("epoch_number"):
                    f.write(f"epoch_number = {n_train}\n")
                elif line.startswith("train_continue ="):
                    f.write("train_continue = False\n")
                elif line.startswith("train_continue_path") or line.startswith("train_continue_epoch"):
                    continue
                else:
                    f.write(line)

        print(f"  {merged_name}: {n_train} train epochs, {n_val} val epochs")

    print("\nDone.")


if __name__ == "__main__":
    main()
