"""
Render log-y histograms from a precomputed JS-divergence jsonl
(no model loading, no GPUs).

Reads the per-token series for js_union / js_intersection / js_student /
js_teacher / iou and produces 5 PNGs into PLOT_DIR.
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


JSONL_PATH = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/js_divergence/amc23_js_topk16_Qwen3-1.7B-Base_Qwen3-4B.jsonl"
PLOT_DIR   = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/js_divergence/amc23_js_topk16_Qwen3-1.7B-Base_Qwen3-4B_plots"

NAME_A = "Qwen3-1.7B-Base"
NAME_B = "Qwen3-4B"
TOP_K  = 16


def plot_distribution(values, xlabel, title, out_path, bins=100):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        print(f"[plot] No valid values for {title}; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    counts, edges = np.histogram(arr, bins=bins)
    proportions   = counts / counts.sum()
    ax.bar(edges[:-1], proportions, width=np.diff(edges), align="edge",
           color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Proportion (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)

    ax.set_yscale("log")
    nonzero = proportions[proportions > 0]
    y_min   = nonzero.min() if nonzero.size else 1.0 / max(arr.size, 1)
    y_low   = 10 ** math.floor(math.log10(y_min))
    ax.set_ylim(y_low, 1.0)

    ax.text(0.98, 0.95,
            f"n={arr.size}\nmean={arr.mean():.4f}\nmedian={np.median(arr):.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    print(f"Reading {JSONL_PATH} ...")
    records = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records.")

    def _flat_valid(key):
        return [v for rec in records for v in rec[key]
                if v is not None and not (isinstance(v, float) and math.isnan(v))]

    all_js_u = _flat_valid("js_union")
    all_js_i = _flat_valid("js_intersection")
    all_js_s = _flat_valid("js_student")
    all_js_t = _flat_valid("js_teacher")
    all_iou  = _flat_valid("iou")

    print(f"  js_union:        {len(all_js_u)} positions")
    print(f"  js_intersection: {len(all_js_i)} positions (NaNs filtered)")
    print(f"  js_student:      {len(all_js_s)} positions")
    print(f"  js_teacher:      {len(all_js_t)} positions")
    print(f"  iou:             {len(all_iou)} positions")

    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    suffix = f"top-k={TOP_K}"

    plot_distribution(
        all_js_u,
        xlabel=f"JS Divergence (nats, support=union, {suffix})",
        title=f"Per-token JS - UNION support\n{NAME_A} (student) vs {NAME_B} (teacher)",
        out_path=Path(PLOT_DIR) / "js_distribution_union.png",
    )
    plot_distribution(
        all_js_i,
        xlabel=f"JS Divergence (nats, support=intersection, {suffix})",
        title=f"Per-token JS - INTERSECTION support (empty-intersection skipped)\n{NAME_A} vs {NAME_B}",
        out_path=Path(PLOT_DIR) / "js_distribution_intersection.png",
    )
    plot_distribution(
        all_js_s,
        xlabel=f"JS Divergence (nats, support=student top-k, {suffix})",
        title=f"Per-token JS - STUDENT top-k support\n{NAME_A} (student) vs {NAME_B} (teacher)",
        out_path=Path(PLOT_DIR) / "js_distribution_student.png",
    )
    plot_distribution(
        all_js_t,
        xlabel=f"JS Divergence (nats, support=teacher top-k, {suffix})",
        title=f"Per-token JS - TEACHER top-k support\n{NAME_A} (student) vs {NAME_B} (teacher)",
        out_path=Path(PLOT_DIR) / "js_distribution_teacher.png",
    )
    plot_distribution(
        all_iou,
        xlabel=f"IoU of top-{TOP_K} token sets",
        title=f"Per-token Top-k Set IoU Distribution\n{NAME_A} vs {NAME_B}",
        out_path=Path(PLOT_DIR) / "iou_distribution.png",
    )


if __name__ == "__main__":
    main()
