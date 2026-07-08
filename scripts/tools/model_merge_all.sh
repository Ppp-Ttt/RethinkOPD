#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mmu_cd_ssd/pengtiantian/projects/OPD/verl}"

# 支持三种传参方式 (优先级: 命令行参数 > 环境变量 > 默认值)
#   1) bash model_merge_all.sh <EXP_DIR> [TARGET_PREFIX]
#   2) EXP_DIR=... TARGET_PREFIX=... bash model_merge_all.sh
#   3) export EXP_DIR=...; export TARGET_PREFIX=...; bash model_merge_all.sh
EXP_DIR="${1:-${EXP_DIR:-/mmu_cd_ssd/pengtiantian/projects/OPD/checkpoint/Qwen3-0.6B-Base_OPD_by_Qwen3-4B-Base-GRPO_FKL_token_reward_direct_DAPO-Math-17k_Qwen3-0.6B-Base_Qwen3-4B-Base-GRPO_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_only_tch-rw_teacher_p-2026-06-24_11-42-09}}"


TARGET_PREFIX="${2:-${TARGET_PREFIX:-Qwen3-0.6B-Base_OPD_by_Qwen3-4B-Base-GRPO_FKL}}"

SKIP_EXISTING="${SKIP_EXISTING:-1}"

# 是否删除非最后一个 step 的 actor 目录
DELETE_OLD_ACTOR="${DELETE_OLD_ACTOR:-1}"

echo "PROJECT_DIR   : $PROJECT_DIR"
echo "EXP_DIR       : $EXP_DIR"
echo "TARGET_PREFIX : $TARGET_PREFIX"
echo "SKIP_EXISTING : $SKIP_EXISTING"
echo "DELETE_OLD_ACTOR: $DELETE_OLD_ACTOR"
echo

cd "$PROJECT_DIR"

shopt -s nullglob

step_dirs=("$EXP_DIR"/global_step_*)

if [ ${#step_dirs[@]} -eq 0 ]; then
    echo "No global_step_* directories found under:"
    echo "$EXP_DIR"
    exit 1
fi

# 对 step 目录按版本号排序
mapfile -t sorted_step_dirs < <(printf "%s\n" "${step_dirs[@]}" | sort -V)

# 最后一个 step，也就是 step 数字最大的目录
last_step_dir="${sorted_step_dirs[-1]}"
last_step_name="$(basename "$last_step_dir")"
last_step_num="${last_step_name#global_step_}"

echo "Last step: $last_step_name"
echo

for step_dir in "${sorted_step_dirs[@]}"; do
    [ -d "$step_dir" ] || continue

    step_name="$(basename "$step_dir")"
    step_num="${step_name#global_step_}"

    actor_dir="$step_dir/actor"
    target_dir="$step_dir/${TARGET_PREFIX}_step${step_num}"

    if [ ! -d "$actor_dir" ]; then
        echo "[SKIP] $step_name: actor directory not found: $actor_dir"
        continue
    fi

    if [ "$SKIP_EXISTING" = "1" ] && [ -d "$target_dir" ]; then
        echo "[SKIP] $step_name: target already exists: $target_dir"
        continue
    fi

    echo "============================================================"
    echo "[MERGE] $step_name"
    echo "actor_dir : $actor_dir"
    echo "target_dir: $target_dir"
    echo "============================================================"

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$target_dir"

    echo "[DONE] $step_name"

    # 转换成功后，删除非最后一个 step 的 actor 目录
    if [ "$DELETE_OLD_ACTOR" = "1" ] && [ "$step_num" != "$last_step_num" ]; then
        echo "[DELETE] Removing old actor directory: $actor_dir"
        rm -rf "$actor_dir"
    else
        echo "[KEEP] Keep actor directory for last step: $actor_dir"
    fi

    echo
done