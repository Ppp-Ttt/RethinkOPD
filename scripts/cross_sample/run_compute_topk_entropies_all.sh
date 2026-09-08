#!/usr/bin/env bash
# Batch-compute top-k entropy files for every JS threshold directory.
# Existing *_top16_entropy.jsonl outputs are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js"
TOP_K=16
LOG_DIR="${RESULT_ROOT}/logs_topk_entropy"

mkdir -p "${LOG_DIR}"

WRAPPER="$(mktemp /tmp/compute_topk_entropies_one.XXXXXX.py)"
trap 'rm -f "${WRAPPER}"' EXIT

cat >"${WRAPPER}" <<'PYEOF'
import multiprocessing as mp
import sys

sys.path.insert(0, sys.argv[1])
import compute_topk_entropies as m

m.INPUT_PATH = sys.argv[2]
m.OUTPUT_PATH = sys.argv[3]
m.TOP_K = int(sys.argv[4])

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    m.main()
PYEOF

shopt -s nullglob

total=0
skipped=0
done_ok=0
failed=0

for dir in "${RESULT_ROOT}"/*_jsth*_topk*/; do
    dir="${dir%/}"
    tag="$(basename "${dir}")"

    for input in "${dir}"/*-MNT*.jsonl; do
        base="$(basename "${input}" .jsonl)"
        # Skip derived files (entropy / prob_diff / iou_mass / grading outputs).
        case "${base}" in
            *_top[0-9]*|*_grading) continue ;;
        esac

        output="${dir}/${base}_top${TOP_K}_entropy.jsonl"
        total=$((total + 1))

        if [[ -f "${output}" ]]; then
            echo "[skip] ${tag}/${base}: output already exists"
            skipped=$((skipped + 1))
            continue
        fi

        stale=("${dir}"/."${base}_top${TOP_K}_entropy.jsonl".gpu*.tmp)
        if (( ${#stale[@]} > 0 )); then
            echo "[skip] ${tag}/${base}: stale tmp files present, clean them first:"
            printf '         %s\n' "${stale[@]}"
            failed=$((failed + 1))
            continue
        fi

        log="${LOG_DIR}/${tag}__${base}.log"
        echo "[run ] ${tag}/${base} -> ${output}"
        echo "       log: ${log}"

        if python "${WRAPPER}" "${SCRIPT_DIR}" "${input}" "${output}" "${TOP_K}" 2>&1 | tee "${log}"; then
            done_ok=$((done_ok + 1))
        else
            echo "[FAIL] ${tag}/${base}: see ${log}"
            failed=$((failed + 1))
        fi
    done
done

echo "========================================================"
echo "candidates: ${total}  computed: ${done_ok}  skipped: ${skipped}  failed: ${failed}"
echo "========================================================"
[[ ${failed} -eq 0 ]]
