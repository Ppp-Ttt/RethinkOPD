cd /mmu_cd_ssd/pengtiantian/projects/OPD/verl
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mmu_cd_ssd/pengtiantian/projects/OPD/checkpoint_rerun0820/1.7B-4B_sparse_rkl_tch_t0.7_token_reward_direct_DAPO-Math-17k_Qwen3-1.7B-Base_Qwen3-4B-Base-GRPO_7168-T_1.0-Tch_0.7-n_4-mbs_64-topk_16-topk_strategy_union-rw_sparse_rkl-2026-09-02_14-18-00/global_step_120/actor \
    --target_dir /mmu_cd_ssd/pengtiantian/projects/OPD/checkpoint_rerun0820/1.7B-4B_sparse_rkl_tch_t0.7_token_reward_direct_DAPO-Math-17k_Qwen3-1.7B-Base_Qwen3-4B-Base-GRPO_7168-T_1.0-Tch_0.7-n_4-mbs_64-topk_16-topk_strategy_union-rw_sparse_rkl-2026-09-02_14-18-00/global_step_120/1.7B-4B_sparse_rkl_tch_t0.7