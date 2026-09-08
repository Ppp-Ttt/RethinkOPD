# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import math
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, top_k=0, student_top_k_ids=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0

                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation and gradient
                    # issues from inplace operations
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        # Note: we don't use inplace_backward here to ensure correct gradients
                        # when both log_probs and topk_log_probs are needed
                        log_probs_all = torch.log_softmax(logits_rmpad, dim=-1)
                        # Gather log_probs for target tokens
                        log_probs = log_probs_all.gather(
                            dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             # Use specific IDs (from rollout)
                             topk_ids = student_top_k_ids
                             if student_top_k_ids.ndim == 3: # (bsz, seqlen, k)
                                 # We are in rmpad mode, but student_top_k_ids is padded 3D tensor
                                 # We need to extract the relevant tokens aligning with input_ids_rmpad_rolled
                                 
                                 # This is tricky because student_top_k_ids is shaped (batch, seq, k)
                                 # and logits_rmpad is (total_nnz, vocab)
                                 # We need to flatten student_top_k_ids to (total_nnz, k) using indices
                                 
                                 # Re-use the indices computed from unpad_input
                                 # indices: (total_nnz,) 
                                 # student_top_k_ids: (batch, seq, k)
                                 
                                 # 1. If student_top_k_ids only covers the response, pad it to match full sequence length
                                 if student_top_k_ids.shape[1] != seqlen:
                                     full_student_top_k_ids = torch.zeros((batch_size, seqlen, top_k), 
                                                                         dtype=student_top_k_ids.dtype, 
                                                                         device=student_top_k_ids.device)
                                     full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                                     student_top_k_ids = full_student_top_k_ids

                                 # 2. Flatten student_top_k_ids to (batch*seq, k)
                                 flat_ids = student_top_k_ids.view(-1, top_k)
                                 
                                 # 3. Select using indices
                                 # Note: indices are from attention_mask, which aligns with how logits_rmpad represents data
                                 topk_ids_rmpad = flat_ids[indices] # (total_nnz, k)
                                 
                                 # If 'student_top_k_ids' in batch has shape (batch, seq_len, k), then:
                                 topk_ids = topk_ids_rmpad
                                 
                             else:
                                 # If it's already flattened? Unlikely.
                                 pass

                        else:
                             # Legacy/Resample behavior
                             _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)

                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if top_k > 0:
                         topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                
                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                
                need_logits = top_k > 0
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        log_probs_all = torch.log_softmax(logits, dim=-1)
                        # Gather log_probs for target tokens (responses)
                        log_probs = log_probs_all.gather(
                            dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             # Ensure shape alignment if needed, but for non-rmpad (bsz, seq, k) should match logits (bsz, seq, vocab) dim 0,1
                        else:
                             _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                        
                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

            return entropy, log_probs, topk_ids, topk_log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability for specific token ids
        Args:
            data (DataProto): a DataProto containing input_ids, attention_mask, position_ids, responses, 
                             and target_ids (batch, response_len, k) in batch
        Returns:
            torch.Tensor: (batch, response_len, k) log probs for target_ids
        """
        # set to eval
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                # We reuse _forward_micro_batch. It returns (entropy, log_probs, topk_ids, topk_log_probs)
                _, _, _, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, 
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # topk_log_probs = topk_log_probs.to("cpu")
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        """Compute the distillation reward (rm_scores) on GPU
        Args:
            data (DataProto): containing all necessary tensors for distillation reward calculation
        Returns:
            DataProto: containing rm_scores and other updated tensors (e.g., union_ids)
        """
        # Set to eval mode for forward passes
        self.actor_module.eval()

        # 1. Extract parameters from meta_info
        top_k = data.meta_info.get("log_prob_top_k", 0)
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")

        # Modes that add a forward-KL term on the top-signal tokens instead of switching
        # branch. They differ only in the per-token routing signal used to rank tokens.
        ADD_FKL_MODES = ("js_add_fkl", "tch_en_add_fkl", "stu_en_add_fkl")
        # Modes that split routed tokens into four student/teacher entropy quadrants and treat
        # the ones in SELECTED_REGION differently.
        REGION_MODES = ("sparse_rkl_add_fkl", "sparse_rkl_switch_fkl")
        ROUTING_MODES = (
            ("js_router", "sparse_rkl", "sparse_rkl_ll_top1", "sparse_fkl", "delta_p_weight")
            + ADD_FKL_MODES
            + REGION_MODES
        )
        ROUTE_SIGNAL_OF_MODE = {
            "js_router": "js",
            "sparse_rkl": "js",
            "sparse_rkl_ll_top1": "js",
            "sparse_rkl_add_fkl": "js",
            "sparse_rkl_switch_fkl": "js",
            "sparse_fkl": "js",
            "delta_p_weight": "js",
            "js_add_fkl": "js",
            "tch_en_add_fkl": "teacher_entropy",
            "stu_en_add_fkl": "student_entropy",
        }
        SUPPORTED_MODES = ("student_p", "teacher_p") + ROUTING_MODES

        if reward_weight_mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unknown reward_weight_mode: {reward_weight_mode}. Supported: {SUPPORTED_MODES}."
            )
        if reward_weight_mode != "student_p":
            # Every non-RKL mode needs both a student-side and a teacher-side probability on a
            # shared support, which only these three strategies provide.
            if strategy not in ("only_stu", "only_tch", "union"):
                raise NotImplementedError(
                    f"reward_weight_mode='{reward_weight_mode}' is only supported with "
                    f"top_k_strategy in ('only_stu', 'only_tch', 'union'), got '{strategy}'."
                )
            # The routing signal is a summary statistic over the K candidates, so it is
            # degenerate unless there are at least two of them.
            min_top_k = 2 if reward_weight_mode in ROUTING_MODES else 1
            if top_k is None or top_k < min_top_k:
                raise ValueError(
                    f"reward_weight_mode='{reward_weight_mode}' requires log_prob_top_k >= "
                    f"{min_top_k}, got log_prob_top_k={top_k}."
                )
        if reward_weight_mode in ("teacher_p", "sparse_fkl", "delta_p_weight") + REGION_MODES and strategy == "only_stu":
            raise ValueError(
                f"reward_weight_mode='{reward_weight_mode}' (FKL) requires the K candidates to come "
                "from the teacher distribution; use top_k_strategy='only_tch' (or union)."
            )

        # Entropy quadrant configuration for the REGION_MODES. Quadrant names are two letters,
        # "<student><teacher>", each 'l' (entropy < ENTROPY_TH) or 'h' (>= ENTROPY_TH).
        VALID_REGIONS = ("hh", "hl", "lh", "ll")
        ENTROPY_TH = float(data.meta_info.get("entropy_th", 0.5))
        selected_region_raw = data.meta_info.get("selected_region", "")
        if isinstance(selected_region_raw, str):
            selected_region = [r.strip().lower() for r in selected_region_raw.split(",") if r.strip()]
        else:
            selected_region = [str(r).strip().lower() for r in selected_region_raw if str(r).strip()]
        if reward_weight_mode in REGION_MODES:
            unknown = [r for r in selected_region if r not in VALID_REGIONS]
            if unknown:
                raise ValueError(
                    f"selected_region contains unknown quadrant(s) {unknown}; "
                    f"valid names are {VALID_REGIONS} ('<student><teacher>', l=low, h=high entropy)."
                )
            if not selected_region:
                raise ValueError(
                    f"reward_weight_mode='{reward_weight_mode}' requires a non-empty selected_region, "
                    f"e.g. selected_region='hh,hl'. Valid names: {VALID_REGIONS}."
                )

        # OPD_THRESHOLD selects which tokens get the forward-KL treatment in the routing modes.
        # "X%"  -> percentile: the top X% of valid response tokens by routing signal.
        # "X"   -> absolute: every token whose routing signal exceeds X.
        threshold_raw = data.meta_info.get("opd_threshold", 0.1)
        if isinstance(threshold_raw, str) and threshold_raw.strip().endswith("%"):
            threshold_mode = "percentile"
            threshold_value = float(threshold_raw.strip()[:-1]) / 100.0
        else:
            threshold_mode = "absolute"
            threshold_value = float(threshold_raw)

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # 2. Compute Student Log Probs on Teacher IDs if needed
        # (This replaces the previous call to compute_log_probs_for_ids in ray_trainer)
        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]
            
            # Select keys for micro-batching
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            
            # We need to pass target_ids to _forward_micro_batch, but since we are micro-batching, 
            # we should split target_ids as well.
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"], 
                                non_tensor_batch_keys=non_tensor_select_keys)
            
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, 
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)
        
        # 3. Compute rm_scores on GPU
        # Move all necessary tensors to GPU (they should already be there if passed from fsdp_workers)
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        S_logp = data.batch["student_top_k_log_probs"].to(device)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device)
        
        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        # Token-level validity mask (B, S): used as the population for percentile JS routing.
        response_mask = data.batch.get("response_mask", None)
        if response_mask is not None:
            response_mask = response_mask.to(device).bool()

        def renormalized_probs(s_logp, t_logp, valid_mask):
            """Student/teacher distributions restricted to the valid candidate support.

            Renormalizing over `valid_mask` makes each token row a proper distribution on the
            truncated support, so divergences between the two are well-defined. This is a
            top-k approximation of the full-vocab quantity, not the exact one.

            Returns (log P_s, P_s, log P_t, P_t), each (B, S, K).
            """
            neg_inf = torch.full_like(s_logp, -float("inf"))
            s_masked = torch.where(valid_mask, s_logp, neg_inf)
            t_masked = torch.where(valid_mask, t_logp, neg_inf)
            s_norm = s_masked - torch.logsumexp(s_masked, dim=-1, keepdim=True)
            t_norm = t_masked - torch.logsumexp(t_masked, dim=-1, keepdim=True)
            return s_norm, torch.exp(s_norm), t_norm, torch.exp(t_norm)

        def route_signal(kind, s_norm, p_s, t_norm, p_t, valid_mask):
            """Per-token (B, S) score ranking how much a token needs the forward-KL treatment."""
            zeros = torch.zeros_like(p_s)
            if kind == "teacher_entropy":
                signal = torch.where(valid_mask, -p_t * t_norm, zeros).sum(dim=-1)
            elif kind == "student_entropy":
                signal = torch.where(valid_mask, -p_s * s_norm, zeros).sum(dim=-1)
            else:  # "js": Jensen-Shannon divergence against the mixture M = 0.5 * (P_s + P_t).
                logM = torch.logsumexp(torch.stack([s_norm, t_norm], dim=0), dim=0) - math.log(2.0)
                kl_s_m = torch.where(valid_mask, p_s * (s_norm - logM), zeros).sum(dim=-1)
                kl_t_m = torch.where(valid_mask, p_t * (t_norm - logM), zeros).sum(dim=-1)
                signal = 0.5 * (kl_s_m + kl_t_m)
            return torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        def threshold_mask(signal, pop_mask):
            """Turn a (B, S) routing signal into the (B, S) bool mask of routed tokens."""
            if threshold_mode == "absolute":
                return signal > threshold_value

            # Percentile: take the top `threshold_value` fraction of the valid response tokens.
            pct = max(0.0, min(1.0, threshold_value))
            n_valid = int(pop_mask.sum().item())
            k = int(round(n_valid * pct))
            if n_valid == 0 or k <= 0:
                return torch.zeros_like(signal, dtype=torch.bool)
            if k >= n_valid:
                return pop_mask
            # k-th largest value as the cutoff; tokens at or above it are routed.
            cutoff = torch.topk(signal[pop_mask], k, largest=True, sorted=False).values.min()
            return (signal >= cutoff) & pop_mask

        # Hard threshold on the teacher/student probability gap for the delta_p_weight mode:
        # only candidates the teacher favours by more than this margin receive the delta term.
        DELTA_TH = float(data.meta_info.get("delta_th", 0.5))
        delta_stats = {}
        region_stats = {}

        # Bound on |rm_scores| applied after the mode-specific reward is formed; 0 disables it.
        distill_reward_clip = float(data.meta_info.get("distill_reward_clip", 1.0))

        def compute_rm_scores(s_logp, t_logp, valid_mask, kl_val, normalize=True):
            """Per-candidate distillation reward for the configured REWARD_WEIGHT_MODE.

            Args:
                s_logp, t_logp: student/teacher log probs (B, S, K) over a shared candidate set.
                valid_mask: (B, S, K) bool, which candidates count.
                kl_val: (B, S, K) per-candidate `log p_s - log p_t`, zeroed outside valid_mask.
                normalize: renormalize over the K candidates. False only for
                    union-intersection, whose two halves are disjoint supports.

            Returns:
                rm_scores (B, S, K): the final reward, ready to be used as the advantage.
                router_mask (B, S) bool: tokens given the forward-KL treatment, or None for
                    the non-routing modes.
            """
            def student_probs():
                """p_s over the valid support -- the reverse-KL weight of standard OPD."""
                logp = torch.where(valid_mask, s_logp, torch.full_like(s_logp, -float("inf")))
                if normalize:
                    logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
                return torch.nan_to_num(torch.exp(logp), nan=0.0, posinf=0.0, neginf=0.0)

            # Standard OPD: reverse KL in policy-gradient form.
            if reward_weight_mode == "student_p":
                return -kl_val * student_probs(), None

            s_norm, p_s, t_norm, p_t = renormalized_probs(s_logp, t_logp, valid_mask)

            # Forward KL, as the gradient of the cross-entropy against the teacher: the reward
            # sums to 1 over the support, so grad_z = p_s^raw - p_t. Subtracting p_s here would
            # make it sum to 0 and cancel the p_s^raw term, which under top-k truncation would
            # stop pushing down the probability mass outside the candidate set.
            # Out-of-support entries are zeroed so they contribute no gradient.
            rm_fkl = torch.nan_to_num(
                torch.where(valid_mask, p_t, torch.zeros_like(p_t)),
                nan=0.0, posinf=0.0, neginf=0.0,
            )

            if reward_weight_mode == "teacher_p":
                return rm_fkl, None

            # Routing modes: rank tokens by a per-token signal, then treat the top ones differently.
            signal = route_signal(
                ROUTE_SIGNAL_OF_MODE[reward_weight_mode], s_norm, p_s, t_norm, p_t, valid_mask
            )
            pop_mask = response_mask if response_mask is not None else valid_mask.any(dim=-1)
            router_mask = threshold_mask(signal, pop_mask.bool())

            if reward_weight_mode == "sparse_fkl":
                # Same reward as teacher_p, but only on the routed (high-JS) tokens.
                return (
                    torch.where(router_mask.unsqueeze(-1), rm_fkl, torch.zeros_like(rm_fkl)),
                    router_mask,
                )

            rm_rkl = -kl_val * student_probs()

            if reward_weight_mode == "sparse_rkl":
                # Reverse KL only on the routed (high-JS) tokens; the rest get no gradient.
                return (
                    torch.where(router_mask.unsqueeze(-1), rm_rkl, torch.zeros_like(rm_rkl)),
                    router_mask,
                )

            if reward_weight_mode == "sparse_rkl_ll_top1":
                # sparse_rkl plus a fixed bonus on the teacher's top-1 candidate, at tokens that
                # are JS-routed AND where both teacher and student are confident (entropy below
                # ENTROPY_TH). The point: on positions the model already finds unambiguous, push
                # the student's mass toward the teacher's favourite candidate in the support.
                TOP1_BONUS = 1.0
                teacher_entropy = torch.where(valid_mask, -p_t * t_norm, torch.zeros_like(p_t)).sum(dim=-1)
                student_entropy = torch.where(valid_mask, -p_s * s_norm, torch.zeros_like(p_s)).sum(dim=-1)
                low_entropy = (teacher_entropy < ENTROPY_TH) & (student_entropy < ENTROPY_TH)
                teacher_top1_idx = torch.argmax(
                    torch.where(valid_mask, t_norm, torch.full_like(t_norm, -float("inf"))),
                    dim=-1,
                )  # (B, S)
                teacher_top1 = torch.zeros_like(rm_rkl).scatter_(
                    -1, teacher_top1_idx.unsqueeze(-1), 1.0
                )
                bonus = torch.where(
                    (low_entropy & router_mask).unsqueeze(-1),
                    TOP1_BONUS * teacher_top1,
                    torch.zeros_like(rm_rkl),
                )
                base = torch.where(router_mask.unsqueeze(-1), rm_rkl, torch.zeros_like(rm_rkl))
                return base + bonus, router_mask

            if reward_weight_mode in REGION_MODES:
                # Split routed tokens by the (student, teacher) entropy quadrant and treat the
                # ones in `selected_region` differently. Entropies are taken over the same
                # renormalized candidate support as the routing signal.
                teacher_entropy = torch.where(valid_mask, -p_t * t_norm, torch.zeros_like(p_t)).sum(dim=-1)
                student_entropy = torch.where(valid_mask, -p_s * s_norm, torch.zeros_like(p_s)).sum(dim=-1)
                teacher_entropy = torch.nan_to_num(teacher_entropy, nan=0.0, posinf=0.0, neginf=0.0)
                student_entropy = torch.nan_to_num(student_entropy, nan=0.0, posinf=0.0, neginf=0.0)
                stu_high = student_entropy >= ENTROPY_TH
                tch_high = teacher_entropy >= ENTROPY_TH
                quadrant = {
                    "hh": stu_high & tch_high,
                    "hl": stu_high & ~tch_high,
                    "lh": ~stu_high & tch_high,
                    "ll": ~stu_high & ~tch_high,
                }
                region_stats.update(quadrant)

                region_mask = torch.zeros_like(stu_high)
                for name in selected_region:
                    region_mask = region_mask | quadrant[name]

                if reward_weight_mode == "sparse_rkl_add_fkl":
                    # Selected quadrants keep reverse KL and get forward KL added on top.
                    routed = rm_rkl + torch.where(
                        region_mask.unsqueeze(-1), rm_fkl, torch.zeros_like(rm_fkl)
                    )
                else:  # sparse_rkl_switch_fkl
                    # Selected quadrants swap reverse KL out for forward KL entirely.
                    routed = torch.where(region_mask.unsqueeze(-1), rm_fkl, rm_rkl)

                return (
                    torch.where(router_mask.unsqueeze(-1), routed, torch.zeros_like(routed)),
                    router_mask,
                )

            if reward_weight_mode == "delta_p_weight":
                # sparse_rkl plus a one-sided term over the candidates the teacher favours by
                # more than DELTA_TH. The hard threshold keeps the term focused on the few
                # candidates where student and teacher genuinely disagree.
                # The -log p_s factor is a detached focal-style weight that up-weights candidates
                # the student currently assigns a low probability. It is capped because it grows
                # without bound as p_s -> 0, which would otherwise let the noisiest candidates at
                # the edge of the top-k support dominate the term.
                delta_gap = torch.where(valid_mask, p_t - p_s, torch.zeros_like(p_t))
                delta_hit = delta_gap > DELTA_TH
                focal_w = (-s_norm).clamp(max=5.0)
                delta = torch.nan_to_num(
                    torch.where(delta_hit, delta_gap * focal_w, torch.zeros_like(p_t)),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                delta_stats["hit"] = delta_hit
                return (
                    torch.where(
                        router_mask.unsqueeze(-1), rm_rkl + delta, torch.zeros_like(rm_rkl)
                    ),
                    router_mask,
                )

            # The FKL branch stands on its own: its gradient is already the cross-entropy
            # gradient, so it must NOT pick up the -kl_val factor that the RKL branch carries.
            if reward_weight_mode == "js_router":
                # Routed tokens switch entirely from reverse to forward KL.
                return torch.where(router_mask.unsqueeze(-1), rm_fkl, rm_rkl), router_mask

            # *_add_fkl: every token keeps reverse KL; routed tokens get forward KL added on top.
            return (
                rm_rkl + torch.where(router_mask.unsqueeze(-1), rm_fkl, torch.zeros_like(rm_fkl)),
                router_mask,
            )

        def compute_divergence_stats(S_logp, T_logp, valid_mask, token_mask):
            """Per-sequence sums of RKL/FKL/JSD over the truncated candidate support.

            Both distributions are renormalized over `valid_mask` so each token row is a proper
            distribution on the (approximate full-vocab) support before any divergence is taken.
            Returns (B,) sums plus a (B,) token count so the driver can form an exact
            token-weighted mean across the whole batch rather than averaging per-rank means.
            """
            neg_inf = torch.full_like(S_logp, -float("inf"))
            S_norm = torch.where(valid_mask, S_logp, neg_inf)
            T_norm = torch.where(valid_mask, T_logp, neg_inf)
            S_norm = S_norm - torch.logsumexp(S_norm, dim=-1, keepdim=True)
            T_norm = T_norm - torch.logsumexp(T_norm, dim=-1, keepdim=True)
            P_s = torch.exp(S_norm)
            P_t = torch.exp(T_norm)

            zeros = torch.zeros_like(P_s)
            # RKL = KL(P_s || P_t), FKL = KL(P_t || P_s)
            rkl = torch.where(valid_mask, P_s * (S_norm - T_norm), zeros).sum(dim=-1)
            fkl = torch.where(valid_mask, P_t * (T_norm - S_norm), zeros).sum(dim=-1)
            # JSD against the mixture M = 0.5 * (P_s + P_t), computed in log-space.
            logM = torch.logsumexp(torch.stack([S_norm, T_norm], dim=0), dim=0) - math.log(2.0)
            kl_s_m = torch.where(valid_mask, P_s * (S_norm - logM), zeros).sum(dim=-1)
            kl_t_m = torch.where(valid_mask, P_t * (T_norm - logM), zeros).sum(dim=-1)
            jsd = 0.5 * (kl_s_m + kl_t_m)

            tok = token_mask.float() if token_mask is not None else torch.ones_like(rkl)
            out = {}
            for name, val in (("rkl", rkl), ("fkl", fkl), ("jsd", jsd)):
                val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                out[f"divergence_{name}_sum"] = (val * tok).sum(dim=-1).detach()
            out["divergence_token_count"] = tok.sum(dim=-1).detach()
            return out

        res_tensors = {}
        normalize = True

        # Each strategy defines the candidate support: which K ids the reward is computed over,
        # and which of them are valid. `compute_rm_scores` then applies the configured mode.
        if strategy == "only_stu":
            S_support, T_support = S_logp, T_on_S
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)

        elif strategy == "only_tch":
            S_support, T_support = S_on_T, T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            res_tensors["union_top_k_ids"] = T_ids

        elif strategy == "intersection":
            S_support, T_support = S_logp, T_on_S
            valid_mask = overlap_mask.bool()

        elif strategy in ("union", "union-intersection"):
            S_support = torch.cat([S_logp, S_on_T], dim=-1)
            T_support = torch.cat([T_on_S, T_logp], dim=-1)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            if strategy == "union":
                # Keep every student candidate; keep teacher candidates that aren't duplicates.
                S_valid = torch.ones_like(S_ids, dtype=torch.bool)
            else:
                # union-intersection: keep only the candidates unique to each side.
                S_valid = ~overlap_mask.bool().to(device)
                normalize = False
            valid_mask = torch.cat([S_valid, ~T_in_S], dim=-1)

            res_tensors["union_top_k_ids"] = torch.cat([S_ids, T_ids], dim=-1)
            res_tensors["union_top_k_log_probs"] = S_support
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T

        else:
            raise ValueError(f"Unknown top_k_strategy: {strategy}")

        kl_val = S_support - T_support
        kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
        rm_scores, router_mask = compute_rm_scores(
            S_support, T_support, valid_mask, kl_val, normalize=normalize
        )

        B = rm_scores.shape[0]
        if response_mask is not None:
            stat_token_mask = response_mask.bool()
        else:
            stat_token_mask = valid_mask.any(dim=-1)

        # Reward clipping. The RKL branch carries a -kl_val = log p_t - log p_s factor, which
        # blows up when the teacher assigns a candidate a near-zero probability. Bounding
        # |reward| keeps a single such candidate from dominating the update.
        if distill_reward_clip > 0:
            # Count over the candidates that actually carry gradient, so the ratio isn't
            # diluted by padding and out-of-support entries.
            clip_cand_mask = valid_mask & stat_token_mask.unsqueeze(-1)
            clipped_hit = (rm_scores.abs() > distill_reward_clip) & clip_cand_mask
            clip_ratio = clipped_hit.float().sum() / clip_cand_mask.float().sum().clamp(min=1.0)
            rm_scores = rm_scores.clamp(min=-distill_reward_clip, max=distill_reward_clip)
            res_tensors["distill_reward_clip_ratio"] = clip_ratio.detach().expand(B).contiguous()

        res_tensors["rm_scores"] = rm_scores

        if strategy != "union-intersection":
            # Skipped for union-intersection: its two halves have disjoint supports, so a
            # shared-support teacher/student divergence is not well-defined.
            res_tensors.update(
                compute_divergence_stats(S_support, T_support, valid_mask, response_mask)
            )

        if router_mask is not None:
            router_stat_mask = stat_token_mask

            fkl_count = (router_mask & router_stat_mask).float().sum()
            total_count = router_stat_mask.float().sum().clamp(min=1.0)

            forward_kl_ratio = fkl_count / total_count
            # Broadcast scalar to a (B,) tensor so DataProto can hold it alongside batched tensors.
            res_tensors["opd_router_forward_kl_ratio"] = forward_kl_ratio.detach().expand(B).contiguous()

            if "hit" in delta_stats:
                # Fraction of valid candidates that clear the hard p_t - p_s threshold, over the
                # same token population as forward_kl_ratio so the two curves are comparable.
                cand_mask = valid_mask & router_stat_mask.unsqueeze(-1)
                delta_ratio = (
                    (delta_stats["hit"] & cand_mask).float().sum()
                    / cand_mask.float().sum().clamp(min=1.0)
                )
                res_tensors["opd_router_delta_th_ratio"] = delta_ratio.detach().expand(B).contiguous()

            if region_stats:
                # Share of the routed tokens falling in each entropy quadrant. The denominator
                # is the routed population, since that is the only place the quadrant changes
                # the reward -- the four ratios therefore sum to 1.
                routed = router_mask & router_stat_mask
                routed_count = routed.float().sum().clamp(min=1.0)
                for name, mask in region_stats.items():
                    ratio = (mask & routed).float().sum() / routed_count
                    res_tensors[f"opd_region_{name}_ratio"] = ratio.detach().expand(B).contiguous()
        return DataProto.from_dict(tensors=res_tensors)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        top_k = data.meta_info.get("top_k", 0)
        print(f"In compute_log_prob, top_k: {top_k}")
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_ids, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, top_k=top_k
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                # entropy = entropy.to("cpu")
                entropy_lst.append(entropy)
            if top_k > 0:
                # topk_ids = topk_ids.to("cpu")
                # topk_log_probs = topk_log_probs.to("cpu")
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        topk_ids_tensor = None
        topk_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor

    # TODO: 
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_last_hidden(self, data: DataProto) -> torch.Tensor:
        """For each sample, return the last-layer hidden state at the last
        non-pad token position (= the last response token).

        Returns:
            torch.Tensor of shape (bsz, hidden_dim), on GPU.
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        last_hidden_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            with torch.no_grad():
                last_hidden = self._forward_last_hidden(micro_batch.batch)
            last_hidden_lst.append(last_hidden)

        last_hidden = torch.concat(last_hidden_lst, dim=0)  # (bsz, hidden_dim)

        if use_dynamic_bsz:
            last_hidden = restore_dynamic_batch(last_hidden, batch_idx_list)

        return last_hidden


    def _forward_last_hidden(self, micro_batch) -> torch.Tensor:
        """Run forward with output_hidden_states=True and extract the hidden
        state at each sample's last non-pad position. Returns (bsz, hidden_dim).
        Mirrors the rmpad/Ulysses-SP handling in `_forward_micro_batch`.
        """
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                pad_size = 0
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )

                # NOTE: do NOT pass fused-kernel extra args here; we need raw hidden_states.
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                )
                # last layer hidden: (1, total_nnz_or_sliced, D) -> (total_nnz_or_sliced, D)
                hidden_rmpad = output.hidden_states[-1].squeeze(0)

                if self.use_ulysses_sp:
                    hidden_rmpad = gather_outputs_and_unpad(
                        hidden_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

                # Pad back to (bsz, seqlen, D)
                full_hidden = pad_input(
                    hidden_states=hidden_rmpad,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                )
                full_hidden = output.hidden_states[-1]  # (bsz, seqlen, D)

            # Last non-pad position per sample == last response token (responses are right-padded).
            last_idx = (attention_mask.sum(dim=-1) - 1).clamp(min=0)  # (bsz,)
            gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, full_hidden.size(-1))
            last_hidden = full_hidden.gather(dim=1, index=gather_idx).squeeze(1)  # (bsz, D)

            return last_hidden


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        if "format_mask" in data.batch.keys():
            select_keys.append("format_mask") # (bsz, 1)
        
        # Include student_top_k_log_probs if present (for top-k distillation)
        if "student_top_k_log_probs" in data.batch.keys():
            select_keys.append("student_top_k_log_probs")

        # Include student_top_k_ids if present (for fixing "apples-to-oranges" bug)
        if "student_top_k_ids" in data.batch.keys():
            select_keys.append("student_top_k_ids")

        # Include union_top_k_ids/log_probs for union strategy
        if "union_top_k_ids" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_ids")
            select_keys.append("union_top_k_ids")
            # now we don't need to store student_top_k_ids and student_top_k_log_probs for union strategy
            if "student_top_k_ids" in select_keys:
                select_keys.remove("student_top_k_ids")

        if "union_top_k_log_probs" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_log_probs")
            select_keys.append("union_top_k_log_probs")
            # now we don't need to store student_top_k_log_probs for union strategy
            if "student_top_k_log_probs" in select_keys:
                select_keys.remove("student_top_k_log_probs")   

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    # Check if we have 3D advantages (top-k sampling case)
                    # If so, we need to recompute top-k log probs for correct gradient
                    if advantages.dim() == 3:
                        top_k = advantages.shape[-1]
                        # For union strategy, use union_top_k_ids; otherwise use student_top_k_ids
                        student_top_k_ids = None
                        if "union_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["union_top_k_ids"]
                        elif "student_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["student_top_k_ids"]

                        entropy, _, _, topk_log_probs = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            top_k=top_k, student_top_k_ids=student_top_k_ids
                        )
                        log_prob_for_loss = topk_log_probs
                        
                    else:
                        _, log_prob, *_ = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        log_prob_for_loss = log_prob

                    format_mask = None
                    if "format_mask" in model_inputs.keys():
                        format_mask = model_inputs["format_mask"]
            

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            print("on_policy")
                            # For on-policy (ppo_epochs=1), use current policy as "old"
                            # log_prob_for_loss is already 3D for top-k case
                            old_log_prob = log_prob_for_loss.detach()
                        else:
                            print("off_policy")
                            # For off-policy, use stored log probs
                            # For 3D top-k case, use stored log probs (union or student)
                            if advantages.dim() == 3:
                                if "union_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["union_top_k_log_probs"]
                                elif "student_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["student_top_k_log_probs"]
                                else:
                                    old_log_prob = model_inputs["old_log_probs"]
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob_for_loss,  # 3D for top-k, 2D otherwise
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        format_mask=format_mask,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
