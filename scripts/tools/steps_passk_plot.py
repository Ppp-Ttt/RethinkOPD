import json
import os
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
# Each path is a grading_results.json written by gen_vllm_grade_steps.py, where
# the top-level shape is {"step20": [task_result, ...], "step40": [...], ...}.
RESULT_PATHS = [
    "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-4B-Base_OPD_by_Qwen3-8B-Base-GRPO/grading_results.json",
    "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-4B-Base_OPD_by_Qwen3-8B-Base-GRPO_JS-ADD-FKL/grading_results.json",
    "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-4B-Base_OPD_by_Qwen3-8B-Base-GRPO_JS-ROUTER_new/grading_results.json",
    # "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-8B-Base-GRPO/grading_results.json",
    # "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-1.7B-OPD_by_Qwen3-4B_insertRKL_unionFKL/grading_results.json",
]
DATASET = "aime24"
N = 128
# Which metric to plot per step. Either "mean" / "best" for mean_score / best_score,
# or an integer K to read pass_at_k["pass@K"].
METRIC = "mean_score" # "mean_score", "best_score", "k"(paa@k)
# Manual starting point shared by every curve (fraction in [0, 1]). Each curve is
# extended back to step=0 with this value. Set to None to disable.
STEP0_ACC = 0.0826

# Horizontal baselines drawn as dashed lines. Keys are labels, values are
# accuracies as fractions in [0, 1].

BASELINE_ACC = {                #  AMC23   AIME24
    # "Qwen3-1.7B-Base":0.0178,   #  0.1453  0.0178
    # "Qwen3-1.7B": 0.1345,       #  0.4504  0.1345
    # "Qwen3-4B":   0.2264,       #  0.6834  0.2264
    # "Qwen3-4B-Base": 0.0826, 
    # "Qwen3-4B-Base-GRPO": 0.2482, 
    # "Qwen3-8B-Base": 0.1156,
    # "Qwen3-8B-Base-GRPO": 0.2375,

    # "Qwen3-1.7B-Base":0.1453,   #  0.1453  0.0178
    # "Qwen3-1.7B": 0.4504,       #  0.4504  0.1345
    # "Qwen3-4B":   0.6834,       #  0.6834  0.2264
    # "Qwen3-4B-Base":0.2732,
    # "Qwen3-8B-Base": 0.4658
    # "Qwen3-4B-Base-GRPO": 0.6516,
    # "Qwen3-8B-Base-GRPO": 0.6828,
}
OUTPUT_DIR = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/steps_passk_plot"
PLOT_NAME = "Qwen3-4B-Base_OPD_by_Qwen3-8B-Base-GRPO"
# ─────────────────────────────────────────────────────────────────────────────


def _step_num(step_key: str) -> int:
    return int(step_key[len("step"):])


def _metric_label(metric) -> str:
    if metric == "mean_score":
        return "mean_score (pass@1 avg)"
    if metric == "best_score":
        return "best_score (pass@N)"
    return f"pass@{metric}"


def _extract_metric(record: dict, metric):
    if metric == "mean_score":
        return record.get("mean_score")
    if metric == "best_score":
        return record.get("best_score")
    passk = record.get("pass_at_k", {})
    return passk.get(f"pass@{metric}")


def load_curve(result_path: str, dataset: str, n: int, metric):
    """Return [(step, score), ...] sorted by step for the requested dataset/n/metric."""
    with open(result_path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        print(f"[WARN] {result_path} is not in per-step dict format; skipping.")
        return []

    points = []
    for step_key, records in data.items():
        if not step_key.startswith("step") or not isinstance(records, list):
            continue
        for record in records:
            hp = record.get("hyperparameters", {})
            if hp.get("task_name") == dataset and str(hp.get("n")) == str(n):
                score = _extract_metric(record, metric)
                if score is not None:
                    points.append((_step_num(step_key), score * 100))
                break
    points.sort(key=lambda p: p[0])
    return points


def plot_steps():
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = 0
    for path in RESULT_PATHS:
        model_name = os.path.basename(os.path.dirname(path))
        curve = load_curve(path, DATASET, N, METRIC)
        if not curve:
            print(f"[WARN] No matching entries for dataset={DATASET} n={N} metric={METRIC} in {path}, skipping.")
            continue
        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        if STEP0_ACC is not None and (not xs or xs[0] != 0):
            xs = [0] + xs
            ys = [STEP0_ACC * 100] + ys
        ax.plot(xs, ys, marker="o", label=model_name)
        plotted += 1

    if plotted == 0:
        print("Nothing to plot.")
        plt.close(fig)
        return

    for i, (label, acc) in enumerate(BASELINE_ACC.items()):
        color = f"C{len(RESULT_PATHS) + i}"
        ax.axhline(y=acc * 100, color=color, linestyle="--", linewidth=1.2, label=f"{label} (baseline)")

    ax.set_xlabel("training step")
    ax.set_ylabel(f"{_metric_label(METRIC)} (%)")
    ax.set_title(f"{_metric_label(METRIC)} across training — {DATASET} (n={N})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metric_tag = METRIC if isinstance(METRIC, str) else f"passk{METRIC}"
    save_path = os.path.join(OUTPUT_DIR, f"{DATASET}_n{N}_{metric_tag}_{PLOT_NAME}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    plot_steps()
