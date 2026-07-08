"""
Test script: for a sample of rollouts, compute how many tokens are needed
to reach various top-p thresholds using the FULL vocabulary (exact).

OOM avoidance strategy:
  - Extract all resp_logits from logits_all then immediately delete logits_all.
  - Process positions in chunks of POSITION_CHUNK (default 32) so the
    temporary (chunk, V) tensors for softmax/sort/cumsum stay ~55 MB
    regardless of sequence length.

Usage:
    python test_topp_pool_size.py
"""

import json
import random

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Config
# ============================================================
MODEL_PATH   = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-4B"
ROLLOUT_PATH = "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-4B/aime24_t0.7_p0.95_n128-MNT15360.jsonl"

SAMPLE_N            = 10
SEED                = 42
GPU_ID              = 0
BATCH_TOKEN_BUDGET  = 32768
APPLY_CHAT_TEMPLATE = True
ENABLE_THINKING     = False
MAX_RESPONSE_TOKENS = 15360

# Number of token positions processed per softmax/sort/cumsum call.
# Memory per chunk ≈ POSITION_CHUNK × vocab_size × 3 × 4 bytes
# = 32 × 151936 × 12 ≈ 55 MB — safe regardless of sequence length.
POSITION_CHUNK = 32

TOP_P_VALUES = [0.90, 0.95, 0.98, 0.99]
POOL_SIZE    = 1000   # the value we want to validate
# ============================================================


def make_batches(seqs, token_budget):
    sorted_seqs = sorted(seqs, key=lambda s: len(s["full_ids"]))
    batches, batch, batch_max_len = [], [], 0
    for seq in sorted_seqs:
        seq_len = len(seq["full_ids"])
        new_max = max(batch_max_len, seq_len)
        if batch and new_max * (len(batch) + 1) > token_budget:
            batches.append(batch)
            batch, batch_max_len = [seq], seq_len
        else:
            batch.append(seq)
            batch_max_len = new_max
    if batch:
        batches.append(batch)
    return batches


def main():
    random.seed(SEED)

    # --- Load rollouts ---
    print(f"Loading rollouts from {ROLLOUT_PATH} ...")
    records = []
    with open(ROLLOUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Total records: {len(records)}")

    sampled = random.sample(records, min(SAMPLE_N, len(records)))
    print(f"Sampled {len(sampled)} rollouts.")

    # --- Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Build sequences ---
    seqs = []
    for rec in sampled:
        if APPLY_CHAT_TEMPLATE:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": rec["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
            )
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        else:
            prompt_ids = tokenizer.encode(rec["prompt"], add_special_tokens=True)

        response_ids = tokenizer.encode(rec["response"], add_special_tokens=False)
        if MAX_RESPONSE_TOKENS:
            response_ids = response_ids[:MAX_RESPONSE_TOKENS]
        if not response_ids:
            continue

        seqs.append({
            "full_ids":   prompt_ids + response_ids,
            "prompt_len": len(prompt_ids),
            "resp_len":   len(response_ids),
        })

    total_resp_tokens = sum(s["resp_len"] for s in seqs)
    print(f"Built {len(seqs)} sequences. Total response tokens: {total_resp_tokens}")

    # --- Load model ---
    device = f"cuda:{GPU_ID}"
    print(f"Loading model on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model.eval()
    print("Model loaded.")

    counts = {p: [] for p in TOP_P_VALUES}

    batches = make_batches(seqs, BATCH_TOKEN_BUDGET)
    print(f"Running {len(batches)} batches...")

    for batch_idx, batch in enumerate(batches):
        print(f"  Batch {batch_idx + 1}/{len(batches)} (size={len(batch)})", flush=True)

        max_len = max(len(s["full_ids"]) for s in batch)
        pad_id  = tokenizer.pad_token_id

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for i, s in enumerate(batch):
            ids   = s["full_ids"]
            start = max_len - len(ids)
            input_ids[i, start:] = torch.tensor(ids, dtype=torch.long, device=device)
            attn_mask[i, start:] = 1

        with torch.no_grad():
            logits_all = model(input_ids=input_ids, attention_mask=attn_mask).logits  # (B, L, V)

        # Step 1: extract resp_logits for every sequence (float32, on GPU)
        #         then immediately free logits_all to reclaim its memory.
        resp_logits_list = []
        for i, s in enumerate(batch):
            offset = max_len - len(s["full_ids"])
            p_len  = s["prompt_len"]
            r_len  = s["resp_len"]
            resp_logits_list.append(
                logits_all[i, offset + p_len : offset + p_len + r_len].float()
            )

        del logits_all, input_ids, attn_mask
        torch.cuda.empty_cache()

        # Step 2: process each sequence position-by-position in chunks
        for s, resp_logits in zip(batch, resp_logits_list):
            r_len = resp_logits.shape[0]

            for chunk_start in range(0, r_len, POSITION_CHUNK):
                chunk = resp_logits[chunk_start : chunk_start + POSITION_CHUNK]  # (C, V)

                probs        = torch.softmax(chunk, dim=-1)
                sorted_probs = torch.sort(probs, dim=-1, descending=True).values
                cumsum       = sorted_probs.cumsum(dim=-1)

                for p_val in TOP_P_VALUES:
                    over       = cumsum >= p_val
                    first_over = over.long().argmax(dim=-1)
                    reached    = over.any(dim=-1)
                    k_needed   = torch.where(
                        reached,
                        first_over + 1,
                        torch.full_like(first_over, chunk.shape[-1]),
                    )
                    counts[p_val].extend(k_needed.cpu().tolist())

                del probs, sorted_probs, cumsum

            del resp_logits

        resp_logits_list.clear()
        torch.cuda.empty_cache()

    # --- Statistics ---
    total_pos = len(counts[TOP_P_VALUES[0]])
    print("\n" + "=" * 70)
    print(f"  Exact full-vocab results  |  POOL_SIZE to validate: {POOL_SIZE}")
    print(f"  POSITION_CHUNK = {POSITION_CHUNK}  (peak sort memory ≈ {POSITION_CHUNK * 151936 * 12 / 1e6:.0f} MB)")
    print("=" * 70)
    print(f"{'top-p':>8}  {'mean':>7}  {'median':>7}  {'p95':>7}  {'p99':>7}  {'p99.9':>7}  {'max':>7}  {f'>={POOL_SIZE}(%)':>10}")
    print("-" * 70)

    for p in TOP_P_VALUES:
        arr    = np.array(counts[p])
        exceed = 100.0 * (arr >= POOL_SIZE).mean()
        print(
            f"{p:>8.2f}  {arr.mean():>7.1f}  {np.median(arr):>7.1f}"
            f"  {np.percentile(arr, 95):>7.1f}  {np.percentile(arr, 99):>7.1f}"
            f"  {np.percentile(arr, 99.9):>7.1f}  {int(arr.max()):>7}  {exceed:>9.3f}%"
        )

    print("=" * 70)
    print(f"\nTotal token positions analyzed: {total_pos}")
    print(f"\nConclusion guide:")
    print(f"  '>={POOL_SIZE}(%)' near 0%  →  POOL_SIZE={POOL_SIZE} is safe.")
    print(f"  '>={POOL_SIZE}(%)' non-trivial (>0.1%)  →  increase POOL_SIZE.")


if __name__ == "__main__":
    main()
