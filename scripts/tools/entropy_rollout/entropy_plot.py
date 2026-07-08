"""
Plot per-token entropy scatter comparing three model entropy files.

FILE_X  (x-axis) : reference model — Qwen3-1.7B-Base
FILE_Y1 (y-axis) : first comparison  — Qwen3-4B
FILE_Y2 (y-axis) : second comparison — Qwen3-1.7B-Base-OPD

Records are matched by (example_id, seed); within each match, token
positions are zipped.  Because total point counts can reach tens of
millions, both series are subsampled to N_SAMPLE points (same random
indices, so x values are shared across series) before plotting.
"""

# ============================================================
# Global parameters — edit here before running
# ============================================================

# X-axis: reference model entropy file
FILE_X  = "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-1.7B-Base/amc23_entropy.jsonl"

# Y-axis series 1
FILE_Y1 = "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-4B/amc23_entropy.jsonl"

# Y-axis series 2
FILE_Y2 = "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-1.7B-Base-OPD_by_Qwen3-4B/amc23_entropy.jsonl"

OUTPUT_PATH = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/amc23_entropy.png"

# Legend labels
LABEL_X  = "Qwen3-1.7B-Base"
LABEL_Y1 = "Qwen3-4B"
LABEL_Y2 = "Qwen3-1.7B-Base-OPD"

# Points sampled per series.  Both series share the same N_SAMPLE indices
# so that x values are identical and the comparison is direct.
# Set to None to plot all points (slow for large datasets).
N_SAMPLE = 500_000

# Random seed for reproducible subsampling
RANDOM_SEED = 42

# Scatter appearance
COLOR_Y1    = "#1f77b4"   # blue
COLOR_Y2    = "#ff7f0e"   # orange
ALPHA       = 0.15
MARKER_SIZE = 1.0         # s parameter in ax.scatter (points²)

# Output figure
FIGSIZE = (8, 8)
DPI     = 150

# ============================================================
# Implementation — no need to edit below
# ============================================================

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_entropy_dict(path):
    """Read a JSONL entropy file → {(example_id, seed): List[float]}."""
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["example_id"], rec.get("seed", 0))
            d[key] = rec["entropy"]
    print(f"  {len(d):,} records  ← {path}")
    return d


def collect_pairs(dx, dy1, dy2):
    """
    Match records by (example_id, seed), zip token positions, and return
    three float32 numpy arrays (x, y1, y2) of equal length.
    Positions where any value is NaN or Inf are dropped.
    """
    common_keys = sorted(set(dx) & set(dy1) & set(dy2))
    print(f"  Common (example_id, seed) pairs: {len(common_keys):,}")

    x_parts, y1_parts, y2_parts = [], [], []
    for key in common_keys:
        ex  = dx[key]
        ey1 = dy1[key]
        ey2 = dy2[key]
        n = min(len(ex), len(ey1), len(ey2))
        x_parts.append(np.array(ex[:n],  dtype=np.float32))
        y1_parts.append(np.array(ey1[:n], dtype=np.float32))
        y2_parts.append(np.array(ey2[:n], dtype=np.float32))

    x  = np.concatenate(x_parts)
    y1 = np.concatenate(y1_parts)
    y2 = np.concatenate(y2_parts)

    valid = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"  Dropped {n_dropped:,} positions with NaN/Inf entropy.")
    x, y1, y2 = x[valid], y1[valid], y2[valid]

    print(f"  Total valid token positions: {len(x):,}")
    return x, y1, y2


def main():
    print("Loading entropy files...")
    dx  = load_entropy_dict(FILE_X)
    dy1 = load_entropy_dict(FILE_Y1)
    dy2 = load_entropy_dict(FILE_Y2)

    print("\nCollecting token-level pairs...")
    x, y1, y2 = collect_pairs(dx, dy1, dy2)
    del dx, dy1, dy2  # free raw dicts

    # Subsample
    total = len(x)
    if N_SAMPLE is not None and total > N_SAMPLE:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(total, size=N_SAMPLE, replace=False)
        idx.sort()  # sorted access is faster for large arrays
        x, y1, y2 = x[idx], y1[idx], y2[idx]
        print(f"\nSubsampled {total:,} → {len(x):,} points (seed={RANDOM_SEED}).")
    else:
        print(f"\nUsing all {total:,} points (no subsampling).")

    # Axis limits: 1st–99th percentile to suppress outliers
    all_vals = np.concatenate([x, y1, y2])
    lo = float(np.percentile(all_vals, 0.5))
    hi = float(np.percentile(all_vals, 99.5))

    print("Plotting...")
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.scatter(
        x, y1,
        s=MARKER_SIZE, alpha=ALPHA, color=COLOR_Y1,
        label=LABEL_Y1, rasterized=True, linewidths=0,
    )
    ax.scatter(
        x, y2,
        s=MARKER_SIZE, alpha=ALPHA, color=COLOR_Y2,
        label=LABEL_Y2, rasterized=True, linewidths=0,
    )

    # y = x diagonal reference
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--",
            linewidth=0.8, alpha=0.5, label="y = x")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Entropy — {LABEL_X}  (nats)", fontsize=12)
    ax.set_ylabel("Entropy  (nats)", fontsize=12)
    ax.set_title("Per-token entropy: model comparison", fontsize=13)
    ax.legend(
        markerscale=12,
        fontsize=10,
        handler_map={},
    )
    ax.set_aspect("equal", adjustable="box")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
