"""
Plot student/teacher entropy at token positions above a JS threshold.

For every token where js_values[i] > JS_THRESHOLD:
  x = student_topk_entropy[i]
  y = teacher_topk_entropy[i]

Edit the global parameters below, then run:
  python plot_entropy_above_js.py
"""

# ============================================================================ #
#                  Global parameters - edit here before running                #
# ============================================================================ #

INPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample_0820/cross_sample_1.7B-4B_SPARSE-RKL20%_step279_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_1.7B-4B_SPARSE-RKL20%_step279_TCH_Qwen3-4B-Base-GRPO_jsth0.1_topk16/amc23_t0.7_n8-MNT7168_top16_entropy.jsonl"
)
OUTPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample_0820/cross_sample_1.7B-4B_SPARSE-RKL20%_step279_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_1.7B-4B_SPARSE-RKL20%_step279_TCH_Qwen3-4B-Base-GRPO_jsth0.1_topk16/entropy_scatter.png"
)

JS_THRESHOLD = 0.1
TOP_K = 16

# None plots every selected token. Set an integer to reproducibly subsample.
MAX_POINTS = None
RANDOM_SEED = 42

POINT_COLOR = "#4198ac"
# Four shades derived from the reference green #56BA77, one per quadrant of the
# (student, teacher) entropy plane split at 0.5.
QUADRANT_COLORS = {
    "low_low": "#55b7e6",    # student <= 0.5, teacher <= 0.5
    "low_high": "#55b7e6",   # student <= 0.5, teacher >  0.5
    "high_low": "#55b7e6",   # student >  0.5, teacher <= 0.5
    "high_high": "#55b7e6",  # student >  0.5, teacher >  0.5
}
DIAGONAL_COLOR = "#C14953"
POINT_SIZE = 2.0
POINT_ALPHA = 0.5
FIGSIZE = (8, 8)
DPI = 300

# Refuse to replace an existing figure unless explicitly enabled here.
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


def load_selected_entropies(path: Path, js_threshold: float):
    student_parts = []
    teacher_parts = []
    total_tokens = 0
    selected_tokens = 0
    route_mismatches = 0
    entropy_top_k_values = set()

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            required = (
                "js_values",
                "router_decisions",
                "student_topk_entropy",
                "teacher_topk_entropy",
            )
            missing = [field for field in required if field not in record]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {missing}")

            js_values = np.asarray(record["js_values"], dtype=np.float32)
            student_entropy = np.asarray(
                record["student_topk_entropy"], dtype=np.float32
            )
            teacher_entropy = np.asarray(
                record["teacher_topk_entropy"], dtype=np.float32
            )
            decisions = record["router_decisions"]

            lengths = {
                len(js_values),
                len(student_entropy),
                len(teacher_entropy),
                len(decisions),
            }
            if len(lengths) != 1:
                raise ValueError(
                    f"{path}:{line_no}: per-token arrays have different lengths"
                )

            total_tokens += len(js_values)
            selected = js_values > js_threshold
            finite = (
                np.isfinite(js_values)
                & np.isfinite(student_entropy)
                & np.isfinite(teacher_entropy)
            )
            selected &= finite

            selected_indices = np.flatnonzero(selected)
            route_mismatches += sum(
                decisions[index] != "teacher" for index in selected_indices
            )
            selected_tokens += len(selected_indices)
            if selected_indices.size:
                student_parts.append(student_entropy[selected_indices])
                teacher_parts.append(teacher_entropy[selected_indices])

            if "entropy_top_k" in record:
                entropy_top_k_values.add(int(record["entropy_top_k"]))

    if not student_parts:
        raise ValueError(f"No finite token positions satisfy JS > {js_threshold}.")
    if entropy_top_k_values and entropy_top_k_values != {TOP_K}:
        raise ValueError(
            f"File entropy_top_k values {sorted(entropy_top_k_values)} "
            f"do not match configured TOP_K={TOP_K}."
        )

    student = np.concatenate(student_parts)
    teacher = np.concatenate(teacher_parts)
    return student, teacher, total_tokens, selected_tokens, route_mismatches


def maybe_subsample(student: np.ndarray, teacher: np.ndarray):
    if MAX_POINTS is None or len(student) <= MAX_POINTS:
        return student, teacher, False
    if MAX_POINTS <= 0:
        raise ValueError("MAX_POINTS must be positive or None.")

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(len(student), size=MAX_POINTS, replace=False)
    return student[indices], teacher[indices], True


def piecewise_rescale(values):
    # Map [0, 0.5] -> [0, 0.5] (identity) and [0.5, 2.0] -> [0.5, 1.0]
    # so that 0.5 lands at the axis midpoint and the two ranges occupy
    # equal widths, splitting the plot into four equal squares.
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    low = values <= 0.5
    out[low] = values[low]
    out[~low] = 0.5 + (values[~low] - 0.5) / 3.0
    return out


def main():
    input_path = Path(INPUT_PATH).resolve()
    output_path = Path(OUTPUT_PATH).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input entropy file does not exist: {input_path}")
    if output_path.exists() and not REPLACE:
        raise FileExistsError(f"Output exists; set REPLACE=True to replace it: {output_path}")

    print("Loading token-level entropy data...")
    student, teacher, total_tokens, selected_tokens, route_mismatches = (
        load_selected_entropies(input_path, JS_THRESHOLD)
    )
    if route_mismatches:
        raise ValueError(
            f"Found {route_mismatches} positions with JS > {JS_THRESHOLD} "
            "whose router_decision is not teacher."
        )

    plotted_student, plotted_teacher, subsampled = maybe_subsample(student, teacher)

    in_range = (
        np.isfinite(plotted_student)
        & np.isfinite(plotted_teacher)
        & (plotted_student >= 0.0)
        & (plotted_student <= 2.0)
        & (plotted_teacher >= 0.0)
        & (plotted_teacher <= 2.0)
    )
    dropped = int((~in_range).sum())
    if dropped:
        print(f"Dropping {dropped:,} points outside [0, 2.0] range.")
    plotted_student = plotted_student[in_range]
    plotted_teacher = plotted_teacher[in_range]

    correlation = float(np.corrcoef(student, teacher)[0, 1]) if len(student) > 1 else float("nan")
    selected_ratio = selected_tokens / total_tokens if total_tokens else 0.0

    print(f"Total response tokens : {total_tokens:,}")
    print(f"JS > {JS_THRESHOLD:<8} : {selected_tokens:,} ({selected_ratio:.2%})")
    print(f"Plotted points        : {len(plotted_student):,}")
    print(f"Pearson correlation   : {correlation:.4f}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_x = piecewise_rescale(plotted_student)
    plot_y = piecewise_rescale(plotted_teacher)
    x_high = plotted_student > 0.5
    y_high = plotted_teacher > 0.5
    quadrant_masks = {
        "low_low": ~x_high & ~y_high,
        "low_high": ~x_high & y_high,
        "high_low": x_high & ~y_high,
        "high_high": x_high & y_high,
    }
    for key, mask in quadrant_masks.items():
        print(f"{key:>10}: {int(mask.sum()):,} points")
        if not mask.any():
            continue
        ax.scatter(
            plot_x[mask],
            plot_y[mask],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=QUADRANT_COLORS[key],
            linewidths=0,
            rasterized=True,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=2, alpha=0.8, label="y = 0.5")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=2, alpha=0.8, label="x = 0.5")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "2.0"])
    ax.set_yticklabels(["0", "0.5", "2.0"])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)
    # ax.set_xlabel(f"Student entropy", fontsize=16)
    # ax.set_ylabel(f"Teacher entropy", fontsize=16)
    # ax.set_title(f"Token entropy where JSD > {JS_THRESHOLD}", fontsize=20)
    ax.tick_params(axis="both", labelsize=28)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
