cd /mmu_cd_ssd/pengtiantian/projects/OPD/verl
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mmu_cd_ssd/pengtiantian/projects/OPD/checkpoint/token_reward_direct_DAPO-Math-17k_Qwen3-1.7B_Qwen3-4B_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_union-rw_reweight_student_p-js0.02-2026-06-13_19-43-54/global_step_279/actor \
    --target_dir /mmu_cd_ssd/pengtiantian/projects/OPD/checkpoint/token_reward_direct_DAPO-Math-17k_Qwen3-1.7B_Qwen3-4B_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_union-rw_reweight_student_p-js0.02-2026-06-13_19-43-54/global_step_279/Qwen3-1.7B-OPD_by_Qwen3-4B_rewight