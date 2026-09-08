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
    "OPD": "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/OPD_top16_prob_diff.jsonl",
    # "JS_ADD_FKL": "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/JS_ADD_FKL_top16_prob_diff.jsonl",
}
OUTPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/Combine_top16_prob_diff_distribution_opd.png"
)
import numpy as np
EDGES = np.array([-1.0, 0.0] + list(np.arange(0.0, 1.0+ 1e-9, 0.2)))

# Selection: teacher-intervened positions with student entropy > S_ENT_THR
# and teacher entropy < T_ENT_THR.
S_ENT_THR = 0.5
T_ENT_THR = 0.5

# Diff lies in [-1, 1]; bin width 0.1 gives 20 bins.
BIN_WIDTH = 0.3

# One (fill, edge) color pair per overlaid distribution, cycled in dict order.
PALETTE = [
    ("#167D77", "#0F5C57"),
    ("#C1666B", "#8F3B40"),
    ("#4C6EF5", "#2B4ACB"),
    ("#E8A33D", "#B57715"),
    ("#7B6CA8", "#54487C"),
]
# Transparency for overlaid bars so overlapping distributions stay visible.
BAR_ALPHA = 0.55
FIGSIZE = (8, 7)
DPI = 300

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

    # Display mapping: compress the two negative bins so each occupies the
    # same screen width as a positive BIN_WIDTH (=0.1) bin.
    #   data [-1.0, -0.5] -> display [-0.2, -0.1]
    #   data [-0.5,  0.0] -> display [-0.1,  0.0]
    #   data [ 0.0,  1.0] -> display [ 0.0,  1.0]  (identity)
    display_edges = np.empty_like(edges)
    display_edges[edges <= -0.5] = -0.2 + (edges[edges <= -0.5] - (-1.0)) * 0.2
    display_edges[(edges > -0.5) & (edges <= 0.0)] = -0.1 + (edges[(edges > -0.5) & (edges <= 0.0)] - (-0.5)) * 0.2
    display_edges[edges > 0.0] = edges[edges > 0.0]

    def to_display(v):
        v = np.asarray(v, dtype=np.float64)
        out = np.empty_like(v)
        m1 = v <= -0.5
        m2 = (v > -0.5) & (v <= 0.0)
        m3 = v > 0.0
        out[m1] = -0.2 + (v[m1] - (-1.0)) * 0.2
        out[m2] = -0.1 + (v[m2] - (-0.5)) * 0.2
        out[m3] = v[m3]
        return out

    display_centers = (display_edges[:-1] + display_edges[1:]) / 2
    display_widths = np.diff(display_edges)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    max_share = 0.0

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
        max_share = max(max_share, shares.max())
        fill, edge = PALETTE[series_index % len(PALETTE)]
        ax.bar(
            display_centers,
            shares,
            width=display_widths * 0.94,
            color=fill,
            edgecolor=edge,
            linewidth=0.7,
            alpha=BAR_ALPHA,
            label=f"{label} (mean={diffs.mean():.3f}, med={np.median(diffs):.3f})",
            zorder=2 + series_index,
        )

    # Reference line at diff = 0 (display coord 0).
    ax.axvline(0.0, color="#C14953", linestyle="--", linewidth=1.1, alpha=0.85)

    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(0, max(max_share * 1.15, 1e-3))
    # Negative side: only -1.0, -0.5, 0.0 ticks, each spaced like a 0.1 bin.
    # Positive side: 0.0 .. 1.0 every 0.2.
    tick_vals = np.array([-1.0, -0.5, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    tick_display = to_display(tick_vals)
    ax.set_xticks(tick_display)
    ax.set_xticklabels([f"{v:.1f}" for v in tick_vals])
    ax.set_xlabel("$p_t-p_s$", fontsize=22)
    ax.set_ylabel("Porportion", fontsize=22)
    ax.tick_params(axis="both", labelsize=16)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.grid(True, axis="y", color="#D7DADD", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    if len(prob_diff_paths) > 1:
        ax.legend(fontsize=13, loc="upper right", framealpha=0.9)
    else:
        (label, _), = prob_diff_paths.items()
        ax.legend(fontsize=16, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()
