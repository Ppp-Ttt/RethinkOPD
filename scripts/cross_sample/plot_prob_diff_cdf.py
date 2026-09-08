"""
Plot the CDF of (teacher_topk_prob - student_topk_prob) over positions
where teacher intervention occurred AND student entropy > 0.5 AND teacher
entropy < 0.5.

x: diff = pt - ps in [-1, 1]
y: cumulative share of selected tokens with diff <= x, drawn as a smooth line.

The entropy and prob-diff metrics live in two separate files produced by
compute_topk_entropies.py and compute_topk_prob_diff.py. They are aligned
record-by-record by (example_id, seed).

Edit the global parameters below, then run:
  python plot_prob_diff_cdf.py
"""

# ============================================================================ #
#                  Global parameters - edit here before running                #
# ============================================================================ #

ENTROPY_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168_top16_entropy.jsonl"
)
PROB_DIFF_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168_top16_prob_diff.jsonl"
)
OUTPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168_top16_prob_diff_cdf.png"
)

# Selection: teacher-intervened positions with student entropy > S_ENT_THR
# and teacher entropy < T_ENT_THR.
S_ENT_THR = 0.5
T_ENT_THR = 0.5

# Number of grid points used to evaluate the CDF; more = smoother curve.
N_GRID = 1000

LINE_COLOR = "#74c476"
FIGSIZE = (8, 5)
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


def main():
    entropy_path = Path(ENTROPY_PATH).resolve()
    prob_diff_path = Path(PROB_DIFF_PATH).resolve()
    output_path = Path(OUTPUT_PATH).resolve()

    if not entropy_path.is_file():
        raise FileNotFoundError(f"Entropy file does not exist: {entropy_path}")
    if not prob_diff_path.is_file():
        raise FileNotFoundError(f"Prob-diff file does not exist: {prob_diff_path}")
    if output_path.exists() and not REPLACE:
        raise FileExistsError(
            f"Output exists; set REPLACE=True to replace it: {output_path}"
        )

    print("Loading entropy and prob-diff records...")
    entropy_records = load_indexed(entropy_path, ("example_id", "seed"))
    prob_diff_records = load_indexed(prob_diff_path, ("example_id", "seed"))

    if set(entropy_records) != set(prob_diff_records):
        missing_in_pd = set(entropy_records) - set(prob_diff_records)
        missing_in_ent = set(prob_diff_records) - set(entropy_records)
        raise ValueError(
            f"Record key mismatch. entropy-only keys: {len(missing_in_ent)}, "
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
        dec_arr = np.array(
            [d == "teacher" for d in decisions], dtype=bool
        )
        total_intervention_tokens += int(dec_arr.sum())

        mask = dec_arr & (s_ent > S_ENT_THR) & (t_ent < T_ENT_THR)
        if mask.any():
            diffs.append(diff[mask])

    if length_mismatches:
        print(f"Warning: skipped {length_mismatches} record(s) with length mismatch")
    if decision_mismatches:
        print(f"Warning: {decision_mismatches} record(s) had differing router_decisions")
    if not diffs:
        raise ValueError(
            "No positions match the selection criteria "
            f"(teacher intervention & student entropy > {S_ENT_THR} "
            f"& teacher entropy < {T_ENT_THR})."
        )

    diffs = np.concatenate(diffs)
    diffs = np.clip(diffs, -1.0, 1.0)
    n_selected = len(diffs)
    selection_rate = n_selected / total_intervention_tokens if total_intervention_tokens else 0.0

    print(f"Records                : {n_records}")
    print(f"Total tokens           : {total_tokens:,}")
    print(f"Teacher-intervention tokens: {total_intervention_tokens:,}")
    print(f"Selected tokens        : {n_selected:,} "
          f"({selection_rate:.2%} of interventions)")
    print(f"Diff mean/median       : {diffs.mean():.4f} / {np.median(diffs):.4f}")
    print(f"Diff min/max           : {diffs.min():.4f} / {diffs.max():.4f}")
    print(f"Diff > 0 share         : {(diffs > 0).mean():.2%}")
    print(f"Diff == 0 share        : {(diffs == 0).mean():.2%}")
    print(f"Diff < 0 share         : {(diffs < 0).mean():.2%}")

    # Empirical CDF evaluated on a fine grid for a smooth-looking curve.
    sorted_diffs = np.sort(diffs)
    grid = np.linspace(-1.0, 1.0, N_GRID)
    cdf = np.searchsorted(sorted_diffs, grid, side="right") / n_selected

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        grid,
        cdf,
        color=LINE_COLOR,
        linewidth=2.0,
    )

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Teacher top-k prob - Student top-k prob")
    ax.set_ylabel("P$(p_t-p_s ≤ x)$")
    ax.set_title(
        f"(JSD > 0.06 & Student entropy > {S_ENT_THR} "
        f"& teacher entropy < {T_ENT_THR})"
    )
    ax.grid(True, color="#D7DADD", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
