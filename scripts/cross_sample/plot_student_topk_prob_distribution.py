"""
Plot the distribution of student-side top-k probability (ps) of the actually-
generated token over positions where teacher intervention occurred AND student
entropy > 0.5 AND teacher entropy < 0.5.

ps is the student's renormalised top-k probability of the actually-generated
token (0 if the token is outside the student's top-k). It is read from the
prob-diff file produced by compute_topk_prob_diff.py; the entropy file
(compute_topk_entropies.py) supplies the selection mask.

Multiple prob-diff files (one per student model) can be overlaid in a single
figure as semi-transparent bars. They all share one ENTROPY_PATH: the entropy
file supplies the entropies and router decisions used to select positions,
while each prob-diff file supplies the ps values for those positions. Records
are aligned by (example_id, seed).

Edit the global parameters below, then run:
  python plot_student_topk_prob_distribution.py
"""

# ============================================================================ #
#                  Global parameters - edit here before running                #
# ============================================================================ #

ENTROPY_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.01_topk16/amc23_t0.7_n8-MNT7168_top16_entropy.jsonl"
)
# Multiple prob-diff files overlaid in one figure. key = legend label,
# value = file path. All files are filtered against the single ENTROPY_PATH
# above (shared student entropy / router decisions), so the selected positions
# are identical across files and only the ps values differ.
PROB_DIFF_PATHS = {
    "Base": "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/Base_top16_prob_diff.jsonl",
    "OPD": "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/OPD_top16_prob_diff.jsonl",
    "JS_ADD_FKL": "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/JS_ADD_FKL_top16_prob_diff.jsonl",
}
OUTPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/Combine_top16_student_topk_prob_distribution.png"
)

# Selection: teacher-intervened positions with student entropy > S_ENT_THR
# and teacher entropy < T_ENT_THR.
S_ENT_THR = 0.5
T_ENT_THR = 0.5

BIN_WIDTH = 0.1

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
FIGSIZE = (8, 8)
DPI = 180

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


def collect_probs(entropy_records, prob_diff_path: Path):
    """Return (probs, stats) for one prob-diff file filtered by entropy_records.

    probs is the 1-D array of student top-k probs (ps) at positions where the
    teacher intervened AND student entropy > S_ENT_THR AND teacher entropy
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

    probs = []
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
        ps = np.asarray(pd["student_topk_prob"], dtype=np.float32)
        pd_decisions = pd["router_decisions"]

        L = len(decisions)
        if len(s_ent) != L or len(t_ent) != L or len(ps) != L:
            length_mismatches += 1
            continue
        if pd_decisions != decisions:
            decision_mismatches += 1

        total_tokens += L
        dec_arr = np.array([d == "teacher" for d in decisions], dtype=bool)
        total_intervention_tokens += int(dec_arr.sum())

        mask = dec_arr & (s_ent > S_ENT_THR) & (t_ent < T_ENT_THR)
        if mask.any():
            probs.append(ps[mask])

    if not probs:
        raise ValueError(
            f"No positions match the selection criteria for {prob_diff_path.name} "
            f"(teacher intervention & student entropy > {S_ENT_THR} "
            f"& teacher entropy < {T_ENT_THR})."
        )

    probs = np.concatenate(probs)
    stats = {
        "n_records": n_records,
        "total_tokens": total_tokens,
        "total_intervention_tokens": total_intervention_tokens,
        "length_mismatches": length_mismatches,
        "decision_mismatches": decision_mismatches,
    }
    return probs, stats


def shares_over_bins(probs, n_bins):
    """Bin probs in [0, 1] onto n_bins and return per-bin proportions."""
    positions = np.clip(probs, 0.0, 1.0)
    bin_width = 1.0 / n_bins
    bin_index = np.minimum(np.floor(positions / bin_width).astype(int), n_bins - 1)
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

    n_bins = int(round(1.0 / BIN_WIDTH))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=FIGSIZE)
    max_share = 0.0

    for series_index, (label, prob_diff_path) in enumerate(prob_diff_paths.items()):
        print(f"\n[{label}] {prob_diff_path.name}")
        probs, stats = collect_probs(entropy_records, prob_diff_path)

        if stats["length_mismatches"]:
            print(f"  Warning: skipped {stats['length_mismatches']} record(s) with length mismatch")
        if stats["decision_mismatches"]:
            print(f"  Warning: {stats['decision_mismatches']} record(s) had differing router_decisions")

        n_selected = len(probs)
        interv = stats["total_intervention_tokens"]
        selection_rate = n_selected / interv if interv else 0.0
        print(f"  Records / tokens      : {stats['n_records']} / {stats['total_tokens']:,}")
        print(f"  Teacher-intervention  : {interv:,}")
        print(f"  Selected tokens       : {n_selected:,} ({selection_rate:.2%} of interventions)")
        print(f"  ps mean/median        : {probs.mean():.4f} / {np.median(probs):.4f}")
        print(f"  ps ==0 / <0.1 / >=0.5 : {(probs == 0).mean():.2%} / {(probs < 0.1).mean():.2%} / {(probs >= 0.5).mean():.2%}")

        shares = shares_over_bins(probs, n_bins)
        max_share = max(max_share, shares.max())
        fill, edge = PALETTE[series_index % len(PALETTE)]
        ax.bar(
            centers,
            shares,
            width=BIN_WIDTH * 0.94,
            color=fill,
            edgecolor=edge,
            linewidth=0.7,
            alpha=BAR_ALPHA,
            label=f"{label} (mean={probs.mean():.3f}, med={np.median(probs):.3f})",
            zorder=2 + series_index,
        )

    ax.set_xlim(-BIN_WIDTH / 2, 1.0 + BIN_WIDTH / 2)
    ax.set_ylim(0, max(max_share * 1.15, 1e-3))
    ax.set_xticks(np.arange(0.0, 1.0 + 1e-9, 0.2))
    ax.set_xticklabels([f"{x:.1f}" for x in np.arange(0.0, 1.0 + 1e-9, 0.2)])
    ax.set_xlabel("Student top-k prob of generated token (ps)", fontsize=16)
    ax.set_ylabel("Proportion", fontsize=16)
    ax.set_title(
        f"Student top-k prob (ps) distribution\n"
        f"(teacher intervention & student entropy > {S_ENT_THR} "
        f"& teacher entropy < {T_ENT_THR})",
        fontsize=18,
    )
    ax.tick_params(axis="both", labelsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.grid(True, axis="y", color="#D7DADD", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(fontsize=13, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()
