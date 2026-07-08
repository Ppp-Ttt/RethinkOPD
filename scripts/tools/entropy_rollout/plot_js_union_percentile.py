"""Plot JS-union percentile curve from a precomputed jsonl."""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


JSONL_PATH = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/js_divergence/amc23_js_topk16_Qwen3-1.7B-Base_Qwen3-4B_plots/amc23_js_topk16_Qwen3-1.7B-Base_Qwen3-4B.jsonl"
OUT_PATH   = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/js_divergence/amc23_js_topk16_Qwen3-1.7B-Base_Qwen3-4B_plots/js_union_percentile_curve.png"

NAME_A = "Qwen3-1.7B-Base"
NAME_B = "Qwen3-4B"
TOP_K  = 16

MARK_PERCENTILES = [90, 95, 99, 100]


def main():
    print(f"Reading {JSONL_PATH} ...")
    vals = []
    n_rec = 0
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_rec += 1
            for v in rec["js_union"]:
                if v is None:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                vals.append(v)
    print(f"Loaded {n_rec} records, {len(vals)} valid js_union positions.")

    arr = np.asarray(vals, dtype=np.float64)

    # Percentile grid 0..100 step 0.1 — fine enough that the tail is visible.
    qs = np.linspace(0.0, 100.0, 1001)
    pcts = np.percentile(arr, qs)

    js_max = float(arr.max())
    js_min = float(arr.min())
    js_mean = float(arr.mean())
    js_median = float(np.median(arr))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(qs, pcts, color="steelblue", linewidth=1.8, label="JS (union) quantile")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, js_max * 1.02)
    ax.set_xlabel("Percentile (%)", fontsize=12)
    ax.set_ylabel("JS Divergence (nats, support=union, top-k=16)", fontsize=12)
    ax.set_title(f"Per-token JS (union) Percentile Curve\n{NAME_A} (student) vs {NAME_B} (teacher)",
                 fontsize=13)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Mark requested percentiles.
    marker_colors = ["#d95f02", "#7570b3", "#1b9e77", "#e7298a"]
    for p, color in zip(MARK_PERCENTILES, marker_colors):
        y = float(np.percentile(arr, p))
        ax.scatter([p], [y], color=color, s=55, zorder=5,
                   edgecolor="black", linewidth=0.5,
                   label=f"p{p} = {y:.4f}")
        # Vertical guide & horizontal label
        ax.vlines(p, 0, y, color=color, linestyle=":", linewidth=0.9, alpha=0.6)
        ax.annotate(f"p{p}\n{y:.4f}",
                    xy=(p, y),
                    xytext=(-6, 8) if p < 100 else (-50, -25),
                    textcoords="offset points",
                    fontsize=9, color=color,
                    ha="right" if p < 100 else "left")

    info = (f"n = {arr.size}\n"
            f"min  = {js_min:.4f}\n"
            f"mean = {js_mean:.4f}\n"
            f"median = {js_median:.4f}\n"
            f"max  = {js_max:.4f}")
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.legend(loc="center right", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved: {OUT_PATH}")
    print("Marked percentiles:")
    for p in MARK_PERCENTILES:
        print(f"  p{p:>3} = {float(np.percentile(arr, p)):.6f}")


if __name__ == "__main__":
    main()
