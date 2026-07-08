"""
Compute per-token entropy for rollouts stored in a .jsonl file.

Three entropy modes
-------------------
"full"   — exact entropy over the entire vocabulary.
"topk"   — approximate entropy renormalised over the top-K tokens.
"topp"   — approximate entropy renormalised over the smallest token
           set whose cumulative probability reaches TOP_P.
           Uses a pool of top-(max(TOP_K*10, 1000)) tokens; a warning
           is emitted when the pool does not cover TOP_P mass.

All modes use a HuggingFace forward pass to obtain logits at each
response token position without re-generating any tokens.
Sequences are distributed across NUM_GPUS GPUs via data parallelism;
each GPU loads an independent model copy and processes its shard.

Output format — one JSON object per line
-----------------------------------------
{
    "example_id": int,
    "seed":       int,
    "response_token_ids": [int, ...],
    "entropy":    [float, ...],   # per response-token entropy (nats)
    "mean_entropy": float,
    "response_length": int
}
"""

# ============================================================
# Global parameters — edit here before running
# ============================================================

MODEL_PATH   = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-4B"
ROLLOUT_PATH = "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-1.7B-Base/amc23_t0.7_p0.95_n128-MNT15360.jsonl"
OUTPUT_PATH  = "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-4B/amc23_entropy.jsonl"

# Whether to reconstruct the chat template prefix used during generation.
APPLY_CHAT_TEMPLATE = True

# enable_thinking passed to apply_chat_template (must match what was used during generation).
ENABLE_THINKING = False

# Entropy computation mode: "full" | "topk" | "topp"
ENTROPY_MODE = "topp"

# TOP_K: number of top tokens used for entropy approximation.
TOP_K = 100

# TOP_P: cumulative probability threshold (only used in "topp" mode).
TOP_P = 0.95

# Maximum number of response tokens to process per rollout entry.
MAX_RESPONSE_TOKENS = 15360

# Model precision: "bfloat16" | "float16" | "float32"
# flash_attention_2 requires bfloat16 or float16.
TORCH_DTYPE = "bfloat16"

# Number of GPUs to use for data-parallel scoring.
# Each GPU loads an independent model copy and processes ~1/NUM_GPUS of sequences.
# Automatically capped to torch.cuda.device_count().
NUM_GPUS = 8

# Maximum total tokens (max_seq_len × batch_size) per forward pass per GPU.
# Controls the trade-off between throughput and peak GPU memory for logits.
#   Logit tensor size ≈ BATCH_TOKEN_BUDGET × vocab_size × dtype_bytes
#   Example (bfloat16, vocab=151936):
#     65536 tokens → ~19 GB logits
#     32768 tokens → ~9.5 GB logits
# A800 80 GB comfortably handles 65536 for a 1-2B model.
BATCH_TOKEN_BUDGET = 65536

# ============================================================
# Implementation — no need to edit below
# ============================================================

import json
import math
import multiprocessing as mp
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------------ #
# Tokenisation helpers
# ------------------------------------------------------------------ #

def build_sequences(records, tokenizer, max_response_tokens):
    """
    Returns a list of dicts:
        full_ids          : prompt_ids + response_ids (list[int])
        prompt_len        : len(prompt_ids)
        response_token_ids: response_ids (list[int])
        example_id, seed
    """
    seqs = []
    skipped = 0
    for rec in records:
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

        if max_response_tokens and len(response_ids) > max_response_tokens:
            response_ids = response_ids[:max_response_tokens]

        if not response_ids:
            skipped += 1
            continue

        seqs.append({
            "example_id":         rec["example_id"],
            "seed":               rec.get("seed", 0),
            "full_ids":           prompt_ids + response_ids,
            "prompt_len":         len(prompt_ids),
            "response_token_ids": response_ids,
        })

    if skipped:
        print(f"[Tokenise] Skipped {skipped} records with empty responses.")
    print(f"[Tokenise] Prepared {len(seqs)} sequences.")
    return seqs


def make_batches(seqs, token_budget):
    """
    Group sequences into batches so that max_seq_len × batch_size ≤ token_budget.
    Sequences are sorted by length so that similar-length sequences are grouped,
    maximising GPU utilisation while bounding peak memory.
    """
    sorted_seqs = sorted(seqs, key=lambda s: len(s["full_ids"]))
    batches = []
    batch = []
    batch_max_len = 0
    for seq in sorted_seqs:
        seq_len = len(seq["full_ids"])
        new_max = max(batch_max_len, seq_len)
        if batch and new_max * (len(batch) + 1) > token_budget:
            batches.append(batch)
            batch = [seq]
            batch_max_len = seq_len
        else:
            batch.append(seq)
            batch_max_len = new_max
    if batch:
        batches.append(batch)
    return batches


# ------------------------------------------------------------------ #
# Entropy computation from logits
# ------------------------------------------------------------------ #

def compute_entropies_from_logits(resp_logits, mode, top_k, top_p):
    """
    Compute per-position entropy from response logits.

    resp_logits : (resp_len, vocab_size) float32 tensor on any device
    Returns     : list[float] — entropy in nats per response token
    """
    resp_len, V = resp_logits.shape

    if mode == "full":
        lp = torch.log_softmax(resp_logits, dim=-1)
        h = -(lp.exp() * lp).sum(dim=-1)
        return h.tolist()

    elif mode == "topk":
        k = min(top_k, V)
        topk_logits, _ = torch.topk(resp_logits, k=k, dim=-1)
        lp = topk_logits - torch.logsumexp(topk_logits, dim=-1, keepdim=True)
        h = -(lp.exp() * lp).sum(dim=-1)
        return h.tolist()

    elif mode == "topp":
        k_pool = min(max(top_k * 10, 1000), V)
        topk_logits, _ = torch.topk(resp_logits, k=k_pool, dim=-1)
        probs = torch.softmax(topk_logits, dim=-1)
        cumsum = probs.cumsum(dim=-1)
        over_p = cumsum > top_p
        has_crossover = over_p.any(dim=-1)
        cutoff_idx = over_p.long().argmax(dim=-1)
        cutoff_idx = torch.where(
            has_crossover, cutoff_idx,
            torch.full_like(cutoff_idx, k_pool - 1),
        )

        topp_warnings = int((~has_crossover).sum().item())
        if topp_warnings > 0:
            warnings.warn(
                f"{topp_warnings}/{resp_len} positions: top-{k_pool} tokens "
                f"did not cover top_p={top_p} mass. Increase TOP_K for better accuracy.",
                stacklevel=2,
            )

        h_list = []
        for pos in range(resp_len):
            c = int(cutoff_idx[pos].item()) + 1
            lp_sub = topk_logits[pos, :c]
            lp_norm = lp_sub - torch.logsumexp(lp_sub, dim=0)
            h_list.append(-(lp_norm.exp() * lp_norm).sum().item())
        return h_list

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ------------------------------------------------------------------ #
# Per-GPU worker
# ------------------------------------------------------------------ #

def score_on_device(rank, seqs_chunk, pad_token_id, tmp_path):
    """
    Load a model copy on cuda:{rank}, run forward passes for seqs_chunk,
    and write JSONL results to tmp_path.

    Requires flash_attn to be installed (pip install flash-attn).
    """
    device = f"cuda:{rank}"
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map[TORCH_DTYPE]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model.eval()

    batches = make_batches(seqs_chunk, BATCH_TOKEN_BUDGET)
    n_batches = len(batches)

    with open(tmp_path, "w", encoding="utf-8") as out_f:
        for batch_idx, batch in enumerate(batches):
            print(
                f"[GPU {rank}] batch {batch_idx + 1}/{n_batches} "
                f"(size={len(batch)})",
                flush=True,
            )

            max_len = max(len(s["full_ids"]) for s in batch)

            input_ids = torch.full(
                (len(batch), max_len), pad_token_id, dtype=torch.long, device=device
            )
            attn_mask = torch.zeros(
                (len(batch), max_len), dtype=torch.long, device=device
            )
            for i, s in enumerate(batch):
                ids = s["full_ids"]
                start = max_len - len(ids)
                input_ids[i, start:] = torch.tensor(ids, dtype=torch.long, device=device)
                attn_mask[i, start:] = 1

            with torch.no_grad():
                logits_all = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                ).logits  # (B, max_len, V)

            for i, s in enumerate(batch):
                offset = max_len - len(s["full_ids"])
                p_len = s["prompt_len"]
                r_len = len(s["response_token_ids"])

                resp_logits = logits_all[
                    i, offset + p_len : offset + p_len + r_len
                ].float()  # (r_len, V)

                entropies = compute_entropies_from_logits(
                    resp_logits, ENTROPY_MODE, TOP_K, TOP_P
                )
                valid = [e for e in entropies if not math.isnan(e)]
                mean_h = sum(valid) / len(valid) if valid else float("nan")

                out_f.write(json.dumps({
                    "example_id":         s["example_id"],
                    "seed":               s["seed"],
                    "response_token_ids": s["response_token_ids"],
                    "entropy":            entropies,
                    "mean_entropy":       mean_h,
                    "response_length":    r_len,
                }, ensure_ascii=False) + "\n")

            del logits_all
            torch.cuda.empty_cache()


# ------------------------------------------------------------------ #
# Main pipeline
# ------------------------------------------------------------------ #

def main():
    print(f"[Config] model        = {MODEL_PATH}")
    print(f"[Config] rollout      = {ROLLOUT_PATH}")
    print(f"[Config] output       = {OUTPUT_PATH}")
    print(f"[Config] mode         = {ENTROPY_MODE}")
    if ENTROPY_MODE in ("topk", "topp"):
        print(f"[Config] top_k        = {TOP_K}")
    if ENTROPY_MODE == "topp":
        print(f"[Config] top_p        = {TOP_P}")
    print(f"[Config] chat_template = {APPLY_CHAT_TEMPLATE}, enable_thinking = {ENABLE_THINKING}")
    print(f"[Config] dtype        = {TORCH_DTYPE}")
    print(f"[Config] num_gpus     = {NUM_GPUS}")
    print(f"[Config] token_budget = {BATCH_TOKEN_BUDGET}")

    # --- Tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Rollout records ---
    print("Loading rollout records...")
    records = []
    with open(ROLLOUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records.")

    # --- Tokenise ---
    seqs = build_sequences(records, tokenizer, MAX_RESPONSE_TOKENS)

    # --- GPU allocation ---
    available_gpus = torch.cuda.device_count()
    n_gpus = min(NUM_GPUS, available_gpus)
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs available.")
    print(f"\n[GPU] Using {n_gpus}/{available_gpus} available GPUs.")

    # Round-robin split keeps sequence lengths distributed evenly across GPUs.
    chunks = [seqs[i::n_gpus] for i in range(n_gpus)]
    for i, chunk in enumerate(chunks):
        print(f"  GPU {i}: {len(chunk)} sequences")

    # --- Temp files (one shard per GPU) ---
    out_dir = Path(OUTPUT_PATH).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_paths = [out_dir / f"_entropy_tmp_gpu{rank}.jsonl" for rank in range(n_gpus)]

    # --- Spawn workers (spawn avoids CUDA-fork issues on Linux) ---
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(n_gpus):
        if not chunks[rank]:
            continue
        p = ctx.Process(
            target=score_on_device,
            args=(rank, chunks[rank], tokenizer.pad_token_id, tmp_paths[rank]),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(
            f"{len(failed)} worker(s) exited with non-zero code: "
            + ", ".join(str(p.exitcode) for p in failed)
        )

    # --- Merge shards ---
    print("\nMerging shards...")
    all_results = []
    for rank in range(n_gpus):
        path = tmp_paths[rank]
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))
        path.unlink()

    # Restore original ordering
    all_results.sort(key=lambda r: (r["example_id"], r["seed"]))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for result in all_results:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(all_results)} records to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
