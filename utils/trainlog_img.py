import argparse
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_train_log(file_path: str) -> Tuple[List[int], Dict[str, List[float]]]:
    epochs: List[int] = []
    metrics: Dict[str, List[float]] = {}

    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    for line in lines:
        if line.startswith("Epoch"):
            # Example: "Epoch 12 : 126"
            parts = line.split()
            if len(parts) >= 2:
                epochs.append(int(parts[1]))
            continue

        if "=" not in line:
            continue

        # Example: "error_rate=0.1,psnr=-20.0,..."
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

    if not epochs:
        raise ValueError("No epoch information found in train log.")

    return epochs, metrics


def plot_metrics(epochs: List[int], metrics: Dict[str, List[float]], output_path: str) -> None:
    valid_items = [(k, v) for k, v in metrics.items() if len(v) == len(epochs)]
    if not valid_items:
        raise ValueError("No valid metrics found. Ensure metric counts match epoch count.")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse train_log.txt and draw all metrics in one figure.")
    parser.add_argument("--log", required=True, help="Path to train_log.txt")
    parser.add_argument(
        "--out",
        default="results/train_metrics.png",
        help="Output image path (default: results/train_metrics.png)",
    )
    args = parser.parse_args()

    epochs, metrics = parse_train_log(args.log)
    plot_metrics(epochs, metrics, args.out)
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()
        