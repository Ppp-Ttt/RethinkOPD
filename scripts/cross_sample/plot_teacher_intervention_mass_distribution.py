"""
Plot the distribution of teacher-side intersection probability mass over
positions where teacher intervention occurred AND student entropy > 0.5 AND
teacher entropy < 0.5.

The entropy and IoU/mass metrics live in two separate files produced by
compute_topk_entropies.py and compute_topk_iou_mass.py. They are aligned
record-by-record by (example_id, seed).

Edit the global parameters below, then run:
  python plot_teacher_intervention_mass_distribution.py
"""

# ============================================================================ #
#                  Global parameters - edit here before running                #
# ============================================================================ #

ENTROPY_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168_top16_entropy.jsonl"
)
IOU_MASS_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168_top16_iou_mass.jsonl"
)
OUTPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168_top16_teacher_intervention_mass_distribution.png"
)

# Selection: teacher-intervened positions with student entropy > S_ENT_THR
# and teacher entropy < T_ENT_THR.
S_ENT_THR = 0.5
T_ENT_THR = 0.5

BIN_WIDTH = 0.05

BAR_COLOR = "#C14953"
EDGE_COLOR = "#8E353D"
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
    iou_path = Path(IOU_MASS_PATH).resolve()
    output_path = Path(OUTPUT_PATH).resolve()

    if not entropy_path.is_file():
        raise FileNotFoundError(f"Entropy file does not exist: {entropy_path}")
    if not iou_path.is_file():
        raise FileNotFoundError(f"IoU/mass file does not exist: {iou_path}")
    if output_path.exists() and not REPLACE:
        raise FileExistsError(
            f"Output exists; set REPLACE=True to replace it: {output_path}"
        )

    print("Loading entropy and IoU/mass records...")
    entropy_records = load_indexed(entropy_path, ("example_id", "seed"))
    iou_records = load_indexed(iou_path, ("example_id", "seed"))

    if set(entropy_records) != set(iou_records):
        missing_in_iou = set(entropy_records) - set(iou_records)
        missing_in_ent = set(iou_records) - set(entropy_records)
        raise ValueError(
            f"Record key mismatch. entropy-only keys: {len(missing_in_ent)}, "
            f"iou-only keys: {len(missing_in_iou)}"
        )

    masses = []
    total_intervention_tokens = 0
    total_tokens = 0
    n_records = 0
    length_mismatches = 0
    decision_mismatches = 0

    for key, ent in entropy_records.items():
        iou = iou_records[key]
        n_records += 1

        decisions = ent["router_decisions"]
        s_ent = np.asarray(ent["student_topk_entropy"], dtype=np.float32)
        t_ent = np.asarray(ent["teacher_topk_entropy"], dtype=np.float32)
        t_mass = np.asarray(iou["teacher_intersection_mass"], dtype=np.float32)
        iou_decisions = iou["router_decisions"]

        L = len(decisions)
        if len(s_ent) != L or len(t_ent) != L or len(t_mass) != L:
            length_mismatches += 1
            continue
        if iou_decisions != decisions:
            decision_mismatches += 1

        total_tokens += L
        dec_arr = np.array(
            [d == "teacher" for d in decisions], dtype=bool
        )
        total_intervention_tokens += int(dec_arr.sum())

        mask = dec_arr & (s_ent > S_ENT_THR) & (t_ent < T_ENT_THR)
        if mask.any():
            masses.append(t_mass[mask])

    if length_mismatches:
        print(f"Warning: skipped {length_mismatches} record(s) with length mismatch")
    if decision_mismatches:
        print(f"Warning: {decision_mismatches} record(s) had differing router_decisions")
    if not masses:
        raise ValueError(
            "No positions match the selection criteria "
            f"(teacher intervention & student entropy > {S_ENT_THR} "
            f"& teacher entropy < {T_ENT_THR})."
        )

    masses = np.concatenate(masses)
    n_selected = len(masses)
    selection_rate = n_selected / total_intervention_tokens if total_intervention_tokens else 0.0

    print(f"Records                : {n_records}")
    print(f"Total tokens           : {total_tokens:,}")
    print(f"Teacher-intervention tokens: {total_intervention_tokens:,}")
    print(f"Selected tokens        : {n_selected:,} "
          f"({selection_rate:.2%} of interventions)")
    print(f"Teacher mass mean/median: {masses.mean():.4f} / {np.median(masses):.4f}")
    print(f"Teacher mass min/max   : {masses.min():.4f} / {masses.max():.4f}")

    positions = np.clip(masses, 0.0, 1.0)
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
    ax.set_xticklabels([f"{c:.2f}" for c in centers])
    ax.set_xlabel("Teacher-side intersection probability mass")
    ax.set_ylabel("Share of selected positions")
    ax.set_title(
        f"Teacher intersection mass distribution\n"
        f"(teacher intervention & student entropy > {S_ENT_THR} "
        f"& teacher entropy < {T_ENT_THR})"
    )
    ax.grid(True, axis="y", color="#D7DADD", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)

    ax.text(
        0.02,
        0.98,
        f"selected = {n_selected:,} / {total_intervention_tokens:,} interventions\n"
        f"mean = {masses.mean():.3f}  median = {np.median(masses):.3f}",
        transform=ax.transAxes,
        ha="left",
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
