"""
Plot the distributions of student top-k prob (ps) and teacher top-k prob (pt)
of the actually-generated token over positions where teacher intervention
occurred AND student entropy < 0.5 AND teacher entropy < 0.5.

ps and pt are read from the prob-diff file produced by compute_topk_prob_diff.py;
the entropy file (compute_topk_entropies.py) supplies the selection mask. Both
files are aligned record-by-record by (example_id, seed). ps and pt are drawn
as overlaid histograms so the two distributions can be compared directly.

Edit the global parameters below, then run:
  python plot_ps_pt_distribution.py
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
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168_top16_ps_pt_low_entropy_distribution.png"
)

# Selection: teacher-intervened positions with student entropy < S_ENT_THR
# and teacher entropy < T_ENT_THR.
S_ENT_THR = 0.5
T_ENT_THR = 0.5

BIN_WIDTH = 0.05

STUDENT_COLOR = "#167D77"
TEACHER_COLOR = "#C14953"
STUDENT_EDGE = "#0F5C57"
TEACHER_EDGE = "#8E353D"
FIGSIZE = (9, 5)
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

    ps_parts = []
    pt_parts = []
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
        pt = np.asarray(pd["teacher_topk_prob"], dtype=np.float32)
        pd_decisions = pd["router_decisions"]

        L = len(decisions)
        if len(s_ent) != L or len(t_ent) != L or len(ps) != L or len(pt) != L:
            length_mismatches += 1
            continue
        if pd_decisions != decisions:
            decision_mismatches += 1

        total_tokens += L
        dec_arr = np.array(
            [d == "teacher" for d in decisions], dtype=bool
        )
        total_intervention_tokens += int(dec_arr.sum())

        # Both entropies low: student and teacher each concentrate on their
        # top candidate(s). Note ps is the student prob of the *actually
        # generated* token, which under teacher intervention is the teacher's
        # top pick - not necessarily the student's.
        mask = dec_arr & (s_ent < S_ENT_THR) & (t_ent < T_ENT_THR)
        if mask.any():
            ps_parts.append(ps[mask])
            pt_parts.append(pt[mask])

    if length_mismatches:
        print(f"Warning: skipped {length_mismatches} record(s) with length mismatch")
    if decision_mismatches:
        print(f"Warning: {decision_mismatches} record(s) had differing router_decisions")
    if not ps_parts:
        raise ValueError(
            "No positions match the selection criteria "
            f"(teacher intervention & student entropy < {S_ENT_THR} "
            f"& teacher entropy < {T_ENT_THR})."
        )

    ps = np.concatenate(ps_parts)
    pt = np.concatenate(pt_parts)
    assert len(ps) == len(pt)
    n_selected = len(ps)
    selection_rate = n_selected / total_intervention_tokens if total_intervention_tokens else 0.0

    print(f"Records                : {n_records}")
    print(f"Total tokens           : {total_tokens:,}")
    print(f"Teacher-intervention tokens: {total_intervention_tokens:,}")
    print(f"Selected tokens        : {n_selected:,} "
          f"({selection_rate:.2%} of interventions)")
    print(f"ps mean/median         : {ps.mean():.4f} / {np.median(ps):.4f}")
    print(f"pt mean/median         : {pt.mean():.4f} / {np.median(pt):.4f}")
    print(f"ps==0                  : {(ps==0).mean():.2%}")
    print(f"pt==0                  : {(pt==0).mean():.2%}")
    print(f"diff > 0 (pt > ps)     : {(pt > ps).mean():.2%}")
    print(f"diff < 0 (pt < ps)     : {(pt < ps).mean():.2%}")

    # Bin both distributions over [0, 1] with BIN_WIDTH.
    n_bins = int(round(1.0 / BIN_WIDTH))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    def bin_shares(values):
        v = np.clip(values, 0.0, 1.0)
        idx = np.minimum(np.floor(v / BIN_WIDTH).astype(int), n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins)
        return counts / counts.sum() if counts.sum() > 0 else counts.astype(float)

    ps_shares = bin_shares(ps)
    pt_shares = bin_shares(pt)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        centers,
        ps_shares,
        width=BIN_WIDTH * 0.94,
        color=STUDENT_COLOR,
        edgecolor=STUDENT_EDGE,
        linewidth=0.7,
        alpha=0.78,
        label=f"ps (student), mean={ps.mean():.3f}",
    )
    ax.bar(
        centers,
        pt_shares,
        width=BIN_WIDTH * 0.94,
        color=TEACHER_COLOR,
        edgecolor=TEACHER_EDGE,
        linewidth=0.7,
        alpha=0.55,
        label=f"pt (teacher), mean={pt.mean():.3f}",
    )

    ax.set_xlim(-BIN_WIDTH / 2, 1.0 + BIN_WIDTH / 2)
    ax.set_ylim(0, max(ps_shares.max(), pt_shares.max()) * 1.15)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{c:.2f}" for c in centers])
    ax.set_xlabel("Top-k prob of generated token")
    ax.set_ylabel("Share of selected positions")
    ax.set_title(
        f"ps / pt distribution\n"
        f"(teacher intervention & student entropy < {S_ENT_THR} "
        f"& teacher entropy < {T_ENT_THR})"
    )
    ax.grid(True, axis="y", color="#D7DADD", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#C8CDD0",
        framealpha=0.9,
    )

    ax.text(
        0.98,
        0.98,
        f"selected = {n_selected:,} / {total_intervention_tokens:,} interventions\n"
        f"ps mean={ps.mean():.3f}  pt mean={pt.mean():.3f}\n"
        f"pt > ps: {(pt > ps).mean():.1%}  pt < ps: {(pt < ps).mean():.1%}",
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
