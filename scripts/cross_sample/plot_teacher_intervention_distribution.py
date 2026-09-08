"""
Plot the distribution of teacher-intervention positions along a trajectory.

For each teacher token in every record's router_decisions, its normalized
position is i / (L - 1), where L is the response length (0 = first generated
token, 1 = last). Tokens are then binned on [0, 1] with width 0.1. The y-axis
reports the share of teacher tokens falling into each bin (sum to 1).

Edit the global parameters below, then run:
  python plot_teacher_intervention_distribution.py
"""

# ============================================================================ #
#                  Global parameters - edit here before running                #
# ============================================================================ #

INPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168_top16_entropy.jsonl"
)
OUTPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168_top16_teacher_intervention_distribution.png"
)

# Right edge of the last bin is inclusive: positions equal to 1.0 fall in the
# final bin. Values slightly above 1.0 (should not happen) are clipped.
BIN_WIDTH = 0.05

BAR_COLOR = "#167D77"
EDGE_COLOR = "#0F5C57"
FIGSIZE = (8, 5)
DPI = 180

# Refuse to replace an existing figure unless explicitly enabled here.
REPLACE =True


# ============================================================================ #
#                                  Implementation                              #
# ============================================================================ #

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opd_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def collect_teacher_positions(path: Path):
    positions = []
    total_teacher_tokens = 0
    total_tokens = 0
    n_records = 0
    records_without_teacher = 0

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            decisions = record.get("router_decisions")
            if decisions is None:
                raise ValueError(f"{path}:{line_no}: missing 'router_decisions'")

            L = len(decisions)
            if L == 0:
                continue
            if L == 1:
                # Single token: define position as 0.0 (avoids div-by-zero).
                norm = np.array([0.0])
            else:
                norm = np.arange(L, dtype=np.float64) / (L - 1)

            teacher_mask = np.array(
                [d == "teacher" for d in decisions], dtype=bool
            )
            n_records += 1
            total_tokens += L
            total_teacher_tokens += int(teacher_mask.sum())
            if not teacher_mask.any():
                records_without_teacher += 1
                continue

            positions.append(norm[teacher_mask])

    if not positions:
        raise ValueError("No teacher tokens found in the input file.")
    return (
        np.concatenate(positions),
        total_teacher_tokens,
        total_tokens,
        n_records,
        records_without_teacher,
    )


def main():
    input_path = Path(INPUT_PATH).resolve()
    output_path = Path(OUTPUT_PATH).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if output_path.exists() and not REPLACE:
        raise FileExistsError(
            f"Output exists; set REPLACE=True to replace it: {output_path}"
        )

    print("Loading router decisions and locating teacher tokens...")
    positions, n_teacher, total_tokens, n_records, records_no_t = (
        collect_teacher_positions(input_path)
    )
    print(f"Records               : {n_records}")
    print(f"Records with no teacher: {records_no_t}")
    print(f"Total tokens          : {total_tokens:,}")
    print(f"Teacher tokens        : {n_teacher:,} "
          f"({n_teacher / total_tokens:.2%} of all tokens)")

    # Clip to [0, 1] and assign each token to a bin of width BIN_WIDTH. The
    # final bin includes position == 1.0 (right edge inclusive); positions are
    # never exactly 1.0 in practice (last token has no decision), but the
    # rule keeps the binning total-conservative.
    positions = np.clip(positions, 0.0, 1.0)
    n_bins = int(round(1.0 / BIN_WIDTH))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.minimum(
        np.floor(positions / BIN_WIDTH).astype(int), n_bins - 1
    )
    counts = np.bincount(bin_index, minlength=n_bins)
    shares = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(
        centers,
        shares,
        width=BIN_WIDTH * 0.94,
        color=BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.7,
    )

    ax.set_xlim(-BIN_WIDTH / 2, 1.0 + BIN_WIDTH / 2)
    ax.set_ylim(0, max(shares.max() * 1.15, 1e-3))
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{c:.1f}" for c in centers])
    ax.set_xlabel("Normalized intervention position in trajectory")
    ax.set_ylabel("Share of teacher tokens")
    ax.set_title("Teacher intervention position distribution")
    ax.grid(True, axis="y", color="#D7DADD", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)

    ax.text(
        0.98,
        0.98,
        f"teacher tokens = {n_teacher:,}\nrecords = {n_records}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#30363B",
        bbox={"facecolor": "white", "edgecolor": "#C8CDD0", "alpha": 0.9, "pad": 5},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
