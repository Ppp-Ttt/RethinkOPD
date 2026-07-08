#!/usr/bin/env bash
set -euo pipefail

MINUTES=480

echo "=========================================="
echo "Will run after ${MINUTES} minute(s)."
echo "Start time: $(date)"
echo "=========================================="

sleep "$((MINUTES * 60))"

echo "=========================================="
echo "Running command now."
echo "Run time: $(date)"
echo "=========================================="

echo "=========================================="
echo "Running model merge"
echo "=========================================="
cd /mmu_cd_ssd/pengtiantian/projects/OPD/scripts/tools
bash model_merge_all.sh

echo "=========================================="
echo "Running eval"
echo "=========================================="
cd /mmu_cd_ssd/pengtiantian/projects/OPD/scripts/val/eval
python gen_vllm_grade_single.py