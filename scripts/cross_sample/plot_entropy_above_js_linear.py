"""
Plot student/teacher entropy at token positions above each directory's JS threshold.

Scans every `*_jsth<TH>_topk<K>` directory under RESULT_ROOT, reads the matching
`*_top<K>_entropy.jsonl`, and writes one scatter plot per threshold. Both axes
are linear over [0, 2.5] (no piecewise rescaling).

For every token where js_values[i] > threshold:
  x = student_topk_entropy[i]
  y = teacher_topk_entropy[i]

Edit the global parameters below, then run:
  python plot_entropy_above_js_linear.py
"""

# ============================================================================ #
#                  Global parameters - edit here before running                #
# ============================================================================ #

RESULT_ROOT = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample_0820/cross_sample_1.7B-4B_SPARSE-RKL20%_step279_TCH_Qwen3-4B-Base-GRPO_js"
)

TOP_K = 16

AXIS_MIN = 0.0
AXIS_MAX = 2.5
AXIS_TICKS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
SPLIT_LINE = 0.5

# None plots every selected token. Set an integer to reproducibly subsample.
MAX_POINTS = None
RANDOM_SEED = 42

POINT_COLOR = "#55b7e6"
POINT_SIZE = 2.0
POINT_ALPHA = 0.5
FIGSIZE = (8, 8)
DPI = 300

# Refuse to replace an existing figure unless explicitly enabled here.
REPLACE = True

# Per-threshold statistics, written to RESULT_ROOT and consumed by the viewer.
SUMMARY_NAME = "scatter_linear_summary.json"


# ============================================================================ #
#                                  Implementation                              #
# ============================================================================ #

import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opd_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR_PATTERN = re.compile(r"_jsth(?P<threshold>[0-9.]+)_topk(?P<top_k>\d+)$")


def discover_jobs(root: Path):
    """Yield (threshold, entropy_file, output_file) for each threshold directory."""
    jobs = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        match = DIR_PATTERN.search(directory.name)
        if not match:
            continue
        if int(match.group("top_k")) != TOP_K:
            continue

        threshold = float(match.group("threshold"))
        entropy_files = sorted(directory.glob(f"*_top{TOP_K}_entropy.jsonl"))
        if not entropy_files:
            print(f"[skip] {directory.name}: no *_top{TOP_K}_entropy.jsonl")
            continue

        for entropy_file in entropy_files:
            stem = entropy_file.name[: -len("_entropy.jsonl")]
            dataset = stem.split("_")[0]
            output = directory / (
                f"{dataset}_top{TOP_K}_entropy_jsgt{match.group('threshold')}"
                f"_scatter_linear.png"
            )
            jobs.append((threshold, entropy_file, output))
    return jobs


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
            selected &= (
                np.isfinite(js_values)
                & np.isfinite(student_entropy)
                & np.isfinite(teacher_entropy)
            )

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
        return student, teacher
    if MAX_POINTS <= 0:
        raise ValueError("MAX_POINTS must be positive or None.")

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(len(student), size=MAX_POINTS, replace=False)
    return student[indices], teacher[indices]


def plot_one(js_threshold: float, input_path: Path, output_path: Path) -> dict:
    print("-" * 72)
    print(f"JS threshold          : {js_threshold}")
    print(f"Input                 : {input_path}")

    if output_path.exists() and not REPLACE:
        raise FileExistsError(
            f"Output exists; set REPLACE=True to replace it: {output_path}"
        )

    student, teacher, total_tokens, selected_tokens, route_mismatches = (
        load_selected_entropies(input_path, js_threshold)
    )
    if route_mismatches:
        raise ValueError(
            f"Found {route_mismatches} positions with JS > {js_threshold} "
            "whose router_decision is not teacher."
        )

    plotted_student, plotted_teacher = maybe_subsample(student, teacher)

    in_range = (
        (plotted_student >= AXIS_MIN)
        & (plotted_student <= AXIS_MAX)
        & (plotted_teacher >= AXIS_MIN)
        & (plotted_teacher <= AXIS_MAX)
    )
    dropped = int((~in_range).sum())
    if dropped:
        print(f"Dropping {dropped:,} points outside [{AXIS_MIN}, {AXIS_MAX}].")
    plotted_student = plotted_student[in_range]
    plotted_teacher = plotted_teacher[in_range]

    correlation = (
        float(np.corrcoef(student, teacher)[0, 1]) if len(student) > 1 else float("nan")
    )
    selected_ratio = selected_tokens / total_tokens if total_tokens else 0.0

    print(f"Total response tokens : {total_tokens:,}")
    print(f"JS > {js_threshold:<12}: {selected_tokens:,} ({selected_ratio:.2%})")
    print(f"Plotted points        : {len(plotted_student):,}")
    print(f"Pearson correlation   : {correlation:.4f}")

    x_high = plotted_student > SPLIT_LINE
    y_high = plotted_teacher > SPLIT_LINE
    quadrants = {
        "low_low": int((~x_high & ~y_high).sum()),
        "low_high": int((~x_high & y_high).sum()),
        "high_low": int((x_high & ~y_high).sum()),
        "high_high": int((x_high & y_high).sum()),
    }
    for name, count in quadrants.items():
        print(f"{name:>10}: {count:,} points")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(
        plotted_student,
        plotted_teacher,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        color=POINT_COLOR,
        linewidths=0,
        rasterized=True,
    )
    ax.axhline(SPLIT_LINE, color="black", linestyle="--", linewidth=2, alpha=0.8)
    ax.axvline(SPLIT_LINE, color="black", linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_xticks(AXIS_TICKS)
    ax.set_yticks(AXIS_TICKS)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)
    ax.tick_params(axis="both", labelsize=28)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")

    return {
        "js_threshold": js_threshold,
        "image": str(output_path),
        "input": str(input_path),
        "total_tokens": total_tokens,
        "selected_tokens": selected_tokens,
        "selected_ratio": selected_ratio,
        "plotted_points": int(len(plotted_student)),
        "dropped_points": dropped,
        "correlation": correlation,
        "mean_student_entropy": float(plotted_student.mean()),
        "mean_teacher_entropy": float(plotted_teacher.mean()),
        "quadrants": quadrants,
    }


def main():
    root = Path(RESULT_ROOT).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {root}")

    jobs = discover_jobs(root)
    if not jobs:
        raise ValueError(f"No threshold directories with entropy files under {root}")

    print(f"Found {len(jobs)} threshold file(s) to plot.")
    summaries = []
    failures = []
    for js_threshold, input_path, output_path in jobs:
        try:
            summaries.append(plot_one(js_threshold, input_path, output_path))
        except Exception as error:
            print(f"[FAIL] {input_path}: {error}")
            failures.append((input_path, error))

    if summaries:
        summaries.sort(key=lambda item: item["js_threshold"])
        summary_path = root / SUMMARY_NAME
        summary_path.write_text(
            json.dumps(
                {
                    "axis_min": AXIS_MIN,
                    "axis_max": AXIS_MAX,
                    "split_line": SPLIT_LINE,
                    "top_k": TOP_K,
                    "entries": summaries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote summary to {summary_path}")

    print("=" * 72)
    print(f"Plotted {len(jobs) - len(failures)}/{len(jobs)} figures.")
    if failures:
        for input_path, error in failures:
            print(f"  failed: {input_path}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
