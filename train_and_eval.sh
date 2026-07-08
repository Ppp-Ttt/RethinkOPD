#!/usr/bin/env bash
set -euo pipefail

########################################
# 用户配置区
########################################

export TOP_K_STRATEGY="union"
export REWARD_WEIGHT_MODE="js_add_fkl"
export JS_THRESHOLD="0.01"
export ACTOR_MODEL_PATH=models/Qwen3-1.7B-Base
export REWARD_MODEL_PATH=models/Qwen3-4B-Base-GRPO
# export REWARD_MODEL_PATH=/mmu_cd_ssd/pengtiantian/projects/OPD/checkpoint/Qwen3-8B-Base-GRPO_DAPO-Math-17k-Processed_Qwen3-8B-Base_Qwen3-4B_7168-T_1.0-Tch_1.0-n_8-mbs_64-topk_0-topk_strategy_union-rw_student_p-2026-06-16_15-22-00/global_step_279/Qwen3-8B-Base-GRPO_step279
export EXP_SHORT_NAME="Qwen3-1.7B-Base_OPD_by_Qwen3-4B-Base-GRPO_JS-ADD-FKL"

export LOG_DIR="./log/$EXP_SHORT_NAME"
mkdir -p "$LOG_DIR"

########################################
# 工具函数
########################################

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

########################################
# 主流程
########################################

echo "=========================================="
echo "Start Training and Evaluating Pipeline"
echo "Start time: $(timestamp)"
echo "Log dir: $LOG_DIR"
echo "=========================================="

echo "=========================================="
echo "Step1: Training"
echo "TOP_K_STRATEGY: $TOP_K_STRATEGY"
echo "REWARD_WEIGHT_MODE: $REWARD_WEIGHT_MODE"
echo "JS_THRESHOLD: $JS_THRESHOLD"
echo "ACTOR_MODEL_PATH: $ACTOR_MODEL_PATH"
echo "REWARD_MODEL_PATH: $REWARD_MODEL_PATH"
echo "EXP_SHORT_NAME: $EXP_SHORT_NAME"
echo "Log dir: $LOG_DIR"
echo "=========================================="

bash on_policy_distillation.sh

if [[ -f "$LOG_DIR/run_info.env" ]]; then
    source "$LOG_DIR/run_info.env"
fi

echo "Returned CKPT_PATH: $CKPT_PATH"

echo "=========================================="
echo "Step2: Evaluating"
echo "=========================================="

cd ./scripts/val/eval
python gen_vllm_grade_single.py \
    --weight_dir "/mmu_cd_ssd/pengtiantian/projects/OPD/$CKPT_PATH/global_step_279/${EXP_SHORT_NAME}_step279" \
    --model_name "$EXP_SHORT_NAME"


echo "=========================================="
echo "Pipeline finished successfully"
echo "End time: $(timestamp)"
echo "Logs saved to: $LOG_DIR"
echo "=========================================="