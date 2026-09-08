"""
Plot the distribution of (teacher_topk_prob - student_topk_prob) over positions
where teacher intervention occurred AND student entropy > 0.5 AND teacher
entropy < 0.5.

Multiple prob-diff files (one per student model) can be overlaid in a single
figure as semi-transparent bars. They all share one ENTROPY_PATH: the entropy
file supplies the student/teacher entropies and router decisions used to select
positions, while each prob-diff file supplies the diff values for those
positions. Records are aligned by (example_id, seed).

Edit the global parameters below, then run:
  python plot_prob_diff_distribution.py
"""

# ============================================================================ #
#                  Global parameters - edit here before running                #
# ============================================================================ #

ENTROPY_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168_top16_entropy.jsonl"
)
# Multiple prob-diff files overlaid in one figure. key = legend label,
# value = file path. All files are filtered against the single ENTROPY_PATH
# above (shared student entropy / router decisions), so the selected positions
# are identical across files and only the diff values differ.
PROB_DIFF_PATHS = {
    "Base": "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/Base_top16_prob_diff.jsonl",
    # "OPD": "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/OPD_top16_prob_diff.jsonl",
    # "JS_ADD_FKL": "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/JS_ADD_FKL_top16_prob_diff.jsonl",
}
OUTPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/Combine_top16_prob_diff_distribution_opd.png"
)
import numpy as np
EDGES = np.array([-1.0, 0.0, 0.2, 0.5, 0.8, 1.0])

# Selection: teacher-intervened positions with student entropy > S_ENT_THR
# and teacher entropy < T_ENT_THR.
S_ENT_THR = 0.5
T_ENT_THR = 0.5

# One color per overlaid distribution, cycled in dict order. Each color is
# used as both the bar fill and edge color for its series.
PALETTE = [
    "#D9D9D9",  # Base
    "#ea9e58",  # OPD
]
# Transparency for overlaid bars so overlapping distributions stay visible.
BAR_ALPHA = 0.7
FIGSIZE = (8, 7)
DPI = 300
# Y-axis piecewise scaling: 0..20% occupies the bottom half of the axes,
# 20..70% occupies the top half, so low-frequency bins stay readable while
# tall bins still fit.
Y_BREAK_LOW = 0.15
Y_BREAK_HIGH = 0.75
YTICKS = np.array([0.0, 0.05, 0.10, 0.15, 0.35, 0.55, 0.75])


def y_scale(v):
    """Map data values in [0, Y_BREAK_HIGH] to display space [0, 1].

    [0, Y_BREAK_LOW] -> [0, 0.5]; [Y_BREAK_LOW, Y_BREAK_HIGH] -> [0.5, 1.0].
    """
    v = np.asarray(v, dtype=float)
    out = np.empty_like(v)
    low_mask = v <= Y_BREAK_LOW
    out[low_mask] = v[low_mask] / Y_BREAK_LOW * 0.5
    out[~low_mask] = 0.5 + (v[~low_mask] - Y_BREAK_LOW) / (Y_BREAK_HIGH - Y_BREAK_LOW) * 0.5
    return out
# Vertical offset (in axes-fraction units) for the increment labels above bars.
LABEL_OFFSET = 0.012
# Annotate each bar with its own share (raw proportion), in addition to the
# increment label. Labels of overlaid series are nudged sideways so they do not
# collide.
SHOW_BAR_VALUES = False
BAR_VALUE_FONTSIZE = 11
BAR_VALUE_XSHIFT = 0.22   # horizontal nudge per series, in display bin units

REPLACE = True


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
import matplotlib.ticker as mticker
import numpy as np


def load_indexed(path: Path, key_fields):
    """Load a JSONL file into {(example_id, seed): record}."""
    indexed = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = tuple(record[field] for field in key_fields)
            if key in indexed:
                raise ValueError(f"{path}:{line_no}: duplicate key {key}")
            indexed[key] = record
    return indexed


def collect_diffs(entropy_records, prob_diff_path: Path):
    """Return (diffs, stats) for one prob-diff file filtered by entropy_records.

    diffs is the 1-D array of teacher-minus-student probs at positions where
    the teacher intervened AND student entropy > S_ENT_THR AND teacher entropy
    < T_ENT_THR. stats carries the printable counters.
    """
    prob_diff_records = load_indexed(prob_diff_path, ("example_id", "seed"))

    if set(entropy_records) != set(prob_diff_records):
        missing_in_pd = set(entropy_records) - set(prob_diff_records)
        missing_in_ent = set(prob_diff_records) - set(entropy_records)
        raise ValueError(
            f"Record key mismatch for {prob_diff_path.name}. "
            f"entropy-only keys: {len(missing_in_ent)}, "
            f"prob_diff-only keys: {len(missing_in_pd)}"
        )

    diffs = []
    total_intervention_tokens = 0
    total_tokens = 0
    n_records = 0
    length_mismatches = 0
    decision_mismatches = 0

    for key, ent in entropy_records.items():
        pd = prob_diff_records[key]
        n_records += 1

        decisions = ent["router_decisions"]
        s_ent = np.asarray(ent["student_topk_entropy"], dtype=np.float32)
        t_ent = np.asarray(ent["teacher_topk_entropy"], dtype=np.float32)
        diff = np.asarray(pd["topk_prob_diff"], dtype=np.float32)
        pd_decisions = pd["router_decisions"]

        L = len(decisions)
        if len(s_ent) != L or len(t_ent) != L or len(diff) != L:
            length_mismatches += 1
            continue
        if pd_decisions != decisions:
            decision_mismatches += 1

        total_tokens += L
        dec_arr = np.array([d == "teacher" for d in decisions], dtype=bool)
        total_intervention_tokens += int(dec_arr.sum())

        mask = dec_arr & (s_ent > S_ENT_THR) & (t_ent < T_ENT_THR)
        if mask.any():
            diffs.append(diff[mask])

    if not diffs:
        raise ValueError(
            f"No positions match the selection criteria for {prob_diff_path.name} "
            f"(teacher intervention & student entropy > {S_ENT_THR} "
            f"& teacher entropy < {T_ENT_THR})."
        )

    diffs = np.concatenate(diffs)
    stats = {
        "n_records": n_records,
        "total_tokens": total_tokens,
        "total_intervention_tokens": total_intervention_tokens,
        "length_mismatches": length_mismatches,
        "decision_mismatches": decision_mismatches,
    }
    return diffs, stats


def shares_over_bins(diffs, edges):
    """Bin diffs onto edges and return per-bin proportions (sum to 1)."""
    diffs_clipped = np.clip(diffs, -1.0, 1.0)
    n_bins = len(edges) - 1
    bin_index = np.searchsorted(edges, diffs_clipped, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    counts = np.bincount(bin_index, minlength=n_bins)
    return counts / counts.sum() if counts.sum() > 0 else counts.astype(float)


def main():
    entropy_path = Path(ENTROPY_PATH).resolve()
    output_path = Path(OUTPUT_PATH).resolve()

    if not entropy_path.is_file():
        raise FileNotFoundError(f"Entropy file does not exist: {entropy_path}")
    if not PROB_DIFF_PATHS:
        raise ValueError("PROB_DIFF_PATHS is empty; provide at least one file.")
    prob_diff_paths = {label: Path(p).resolve() for label, p in PROB_DIFF_PATHS.items()}
    for label, path in prob_diff_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Prob-diff file does not exist ({label}): {path}")
    if output_path.exists() and not REPLACE:
        raise FileExistsError(
            f"Output exists; set REPLACE=True to replace it: {output_path}"
        )

    print("Loading entropy records...")
    entropy_records = load_indexed(entropy_path, ("example_id", "seed"))

    # Bin edges over [-1, 1]: two coarse negative bins [-1,-0.5], [-0.5,0];
    # positive side keeps BIN_WIDTH-wide bins. Shared across all files.
    edges = EDGES

    # Display mapping: each bin occupies the same screen width on the x-axis,
    # so bars are visually equal-width even though their data ranges differ.
    # Bins cover data ranges of unequal width; we map each bin to a unit-width
    # slot in display space.
    n_bins = len(edges) - 1
    display_edges = np.arange(n_bins + 1).astype(float)
    bin_index_to_display_center = (display_edges[:-1] + display_edges[1:]) / 2
    # Equal-width bars (slightly narrower to avoid touching).
    bar_width = 0.94

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Per-bin shares for each series; needed to compute increments.
    series_shares = {}

    for series_index, (label, prob_diff_path) in enumerate(prob_diff_paths.items()):
        print(f"\n[{label}] {prob_diff_path.name}")
        diffs, stats = collect_diffs(entropy_records, prob_diff_path)

        if stats["length_mismatches"]:
            print(f"  Warning: skipped {stats['length_mismatches']} record(s) with length mismatch")
        if stats["decision_mismatches"]:
            print(f"  Warning: {stats['decision_mismatches']} record(s) had differing router_decisions")

        n_selected = len(diffs)
        interv = stats["total_intervention_tokens"]
        selection_rate = n_selected / interv if interv else 0.0
        print(f"  Records / tokens      : {stats['n_records']} / {stats['total_tokens']:,}")
        print(f"  Teacher-intervention  : {interv:,}")
        print(f"  Selected tokens       : {n_selected:,} ({selection_rate:.2%} of interventions)")
        print(f"  Diff mean/median      : {diffs.mean():.4f} / {np.median(diffs):.4f}")
        print(f"  Diff >0 / ==0 / <0    : {(diffs > 0).mean():.2%} / {(diffs == 0).mean():.2%} / {(diffs < 0).mean():.2%}")

        shares = shares_over_bins(diffs, edges)
        series_shares[label] = shares
        color = PALETTE[series_index % len(PALETTE)]
        ax.bar(
            bin_index_to_display_center,
            y_scale(shares),
            width=0.75,
            color=color,
            edgecolor="#1A1A1A",
            linewidth=3,
            alpha=BAR_ALPHA,
            label=label,
            zorder=2 + series_index,
        )

    # Annotate每根柱子自身的占比 (原始值)。
    if SHOW_BAR_VALUES:
        n_series = len(series_shares)
        for series_index, (label, shares) in enumerate(series_shares.items()):
            color = PALETTE[series_index % len(PALETTE)]
            x_shift = (series_index - (n_series - 1) / 2) * BAR_VALUE_XSHIFT
            for i, share in enumerate(shares):
                ax.annotate(
                    f"{share * 100:.1f}%",
                    xy=(bin_index_to_display_center[i] + x_shift, float(y_scale(share))),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=BAR_VALUE_FONTSIZE,
                    color=color,
                    zorder=11,
                )

    # Reference line at the boundary between the negative bin and the first
    # positive bin (display coord 1.0).
    ax.axvline(1.0, color="#C14953", linestyle="--", linewidth=1.1, alpha=0.85)

    ax.set_xlim(0, n_bins)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(y_scale(YTICKS))
    # Show tick labels as the original data bin edges: -1.0, 0.0, 0.2, ... 1.0.
    tick_display = display_edges
    tick_labels = [f"{v:.1f}" for v in edges]
    ax.set_xticks(tick_display)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("$p_t-p_s$", fontsize=22)
    ax.set_ylabel("Porportion", fontsize=22)
    ax.tick_params(axis="both", labelsize=24)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{np.interp(v, [0, 0.5, 1.0], [0, Y_BREAK_LOW, Y_BREAK_HIGH]) * 100:.0f}%")
    )
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)
        spine.set_color("#1A1A1A")
    ax.set_box_aspect(1)

    if len(prob_diff_paths) > 1:
        ax.legend(fontsize=13, loc="upper right", framealpha=0.9)
    else:
        ax.legend(fontsize=16, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()
