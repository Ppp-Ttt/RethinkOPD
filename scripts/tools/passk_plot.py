import json
import os
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
# Each entry is (grading_results.json path, step). The grading_results.json is
# the per-step dict format ({"step{step}": [task_result, ...]}). For baseline
# (untrained) models graded by gen_vllm_grade.py, use step=0. For trained
# checkpoints graded by gen_vllm_grade_steps.py, pick the step you want to plot
# (e.g. the final step).
RESULT_PATHS = [
    # ("/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-1.7B-Base/grading_results.json", 0),
    ("/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-1.7B-Base-OPD_by_Qwen3-4B-Base-GRPO/grading_results.json", 279),
    ("/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-1.7B-Base_OPD_by_Qwen3-4B-Base-GRPO_FKL_new/grading_results.json", 279),
    ("/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-1.7B-Base_OPD_by_Qwen3-4B-Base-GRPO_JS-ADD-FKL/grading_results.json", 279),
    ("/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-1.7B-Base_OPD_by_Qwen3-4B-Base-GRPO_JS-ROUTER_new/grading_results.json", 279),
    # ("/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-1.7B-OPD_by_Qwen3-4B_insertRKL_unionFKL/grading_results.json", 279),
]
DATASET = "amc23"
N = 128
OUTPUT_DIR = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/passk_plot"
PLOT_NAME = "Qwen3-1.7B-Base-OPD_by_Qwen3-4B-Base-GRPO_passk"
# ─────────────────────────────────────────────────────────────────────────────


def load_passk(result_path: str, step: int, dataset: str, n: int) -> dict | None:
    with open(result_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get(f"step{step}")
        if not isinstance(records, list):
            return None
    elif isinstance(data, list):
        # Legacy flat-list format — only meaningful for step=0.
        if step != 0:
            return None
        records = data
    else:
        return None

    for record in records:
        hp = record.get("hyperparameters", {})
        if hp.get("task_name") == dataset and str(hp.get("n")) == str(n):
            return record.get("pass_at_k")
    return None


def plot_passk():
    fig, ax = plt.subplots(figsize=(10, 6))

    for path, step in RESULT_PATHS:
        model_name = os.path.basename(os.path.dirname(path))
        label = f"{model_name} (step={step})"
        passk = load_passk(path, step, DATASET, N)
        if passk is None:
            print(f"[WARN] No entry found for dataset={DATASET} n={N} step={step} in {path}, skipping.")
            continue

        ks = sorted(passk.keys(), key=lambda x: int(x.split("@")[1]))
        x = [int(k.split("@")[1]) for k in ks]
        y = [passk[k] * 100 for k in ks]

        ax.plot(x, y, marker="o", label=label)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: str(int(v))))
    ax.set_xlabel("k")
    ax.set_ylabel("Pass@k (%)")
    ax.set_title(f"Pass@k — {DATASET} (n={N})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{DATASET}_n{N}_{PLOT_NAME}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    plot_passk()
