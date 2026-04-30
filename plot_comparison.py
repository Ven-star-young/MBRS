#!/usr/bin/env python3
"""Generate overlaid comparison plots for Experiment 2 and Experiment 3."""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extend_to_100 import parse_train_log

RESULTS = "results"

METRIC_NAMES = {
    "error_rate": "BER (error_rate)",
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "g_loss": "G Loss",
    "g_loss_on_discriminator": "G Loss (Discriminator)",
    "g_loss_on_encoder": "G Loss (Encoder)",
    "g_loss_on_decoder": "G Loss (Decoder)",
    "d_cover_loss": "D Loss (Cover)",
    "d_encoded_loss": "D Loss (Encoded)",
}


def load_data(folder, log_name):
    path = os.path.join(folder, log_name)
    epochs, metrics = parse_train_log(path)
    if "psnr" in metrics:
        metrics["psnr"] = [abs(v) for v in metrics["psnr"]]
    return epochs, metrics


def plot_comparison(folder_a, folder_b, label_a, label_b, color_a, color_b,
                    title, split_name, log_name, out_name):
    """Plot two experiments' metrics overlaid on shared axes."""
    e_a, m_a = load_data(folder_a, log_name)
    e_b, m_b = load_data(folder_b, log_name)

    shared_keys = [k for k in m_a.keys() if k in m_b]
    valid = [(k, m_a[k], m_b[k]) for k in shared_keys
             if len(m_a[k]) == len(e_a) and len(m_b[k]) == len(e_b)]
    if not valid:
        print(f"  No shared metrics for {title} {split_name}")
        return

    n = len(valid)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.2 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (key, v_a, v_b) in enumerate(valid):
        ax = axes_flat[idx]
        ax.plot(e_a, v_a, alpha=0.75, linewidth=1.0, color=color_a, label=label_a)
        ax.plot(e_b, v_b, alpha=0.75, linewidth=1.0, color=color_b, label=label_b)
        ax.set_title(METRIC_NAMES.get(key, key), fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"{title} — {split_name} Metrics Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = os.path.join(RESULTS, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    # Experiment 2: No Diffusion vs With Diffusion (both have Crop)
    no_diff = os.path.join(RESULTS, "experiment1_baseline_with_crop_merged")
    with_diff = os.path.join(RESULTS, "experiment2_crop_diffusion_merged")

    print("Experiment 2: No Diffusion vs With Diffusion (+ Crop)")
    for split, log_name, out_name in [
        ("Training", "train_log.txt", "experiment2_train_comparison.png"),
        ("Validation", "val_log.txt", "experiment2_val_comparison.png"),
    ]:
        plot_comparison(
            no_diff, with_diff,
            "No Diffusion", "With Diffusion",
            "#2196F3", "#F44336",
            "Experiment 2", split, log_name, out_name,
        )

    # Experiment 3: SE Block (Exp1) vs Self-Attention (Exp3) (both no Crop, no Diffusion)
    se_block = os.path.join(RESULTS, "experiment1_baseline_merged")
    self_attn = os.path.join(RESULTS, "experiment3_SelfAttn_merged")

    print("\nExperiment 3: SE Block vs Self-Attention")
    for split, log_name, out_name in [
        ("Training", "train_log.txt", "experiment3_train_comparison.png"),
        ("Validation", "val_log.txt", "experiment3_val_comparison.png"),
    ]:
        plot_comparison(
            se_block, self_attn,
            "SE Block", "Self-Attention",
            "#4CAF50", "#FF9800",
            "Experiment 3", split, log_name, out_name,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
