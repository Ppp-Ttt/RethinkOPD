"""
Compute per-token Jensen-Shannon (JS) divergence between two models
for rollouts stored in a .jsonl file.

Distribution approximation
--------------------------
For each token position the top-k set is computed for each model
(student = Model A, teacher = Model B) using a fixed TOP_K (default 16).

JS divergence is then computed on FOUR different support sets:

  union        — S_A ∪ S_B   (both models' top-k tokens combined)
  intersection — S_A ∩ S_B   (only tokens both models rank in top-k;
                              positions with empty intersection are skipped)
  student      — S_A         (student's top-k tokens)
  teacher      — S_B         (teacher's top-k tokens)

For each support set S, both distributions are renormalised on S
(equivalent to taking softmax inside S), then:

  JS(P_A || P_B) = 0.5*KL(P_A || M) + 0.5*KL(P_B || M),  M = 0.5*(P_A+P_B)

Additional per-position metric
------------------------------
  iou = |S_A ∩ S_B| / |S_A ∪ S_B|    (top-k token-set overlap)

Memory layout
-------------
Both models are loaded on each worker GPU.  For each batch:
  1. Forward pass A  →  logits_all_A  (bfloat16, kept on GPU)
  2. Forward pass B  →  logits_all_B  (bfloat16, kept on GPU)
  3. Per-sequence, per POSITION_CHUNK: extract float32 slice, compute JS
     on all four support sets, accumulate scalars, discard tensors.
  4. Delete both logits_all tensors, empty cache.

Output format (one JSON line per rollout)
-----------------------------------------
{
  "example_id": int,
  "seed":       int,
  "response_token_ids": [int, ...],
  "js_union":          [float, ...],   "mean_js_union":          float,
  "js_intersection":   [float, ...],   "mean_js_intersection":   float,
  "js_student":        [float, ...],   "mean_js_student":        float,
  "js_teacher":        [float, ...],   "mean_js_teacher":        float,
  "iou":               [float, ...],   "mean_iou":               float,
  "empty_intersection_ratio": float,   # fraction of positions with |S_A ∩ S_B|=0
  "response_length": int
}
"""

# ============================================================
# Global parameters — edit here before running
# ============================================================

MODEL_PATH_A = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-1.7B-Base-OPD"   # student
MODEL_PATH_B = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-4B-Base-GRPO"          # teacher

ROLLOUT_PATH = "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output/Qwen3-1.7B-Base/amc23_t0.7_p0.95_n128-MNT15360.jsonl"
OUTPUT_PATH  = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/js_divergence/amc23_js_topk16_Qwen3-1.7B-Base-OPD_Qwen3-4B-Base-GRPO/js.jsonl"
SUMMARY_PATH = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/js_divergence/amc23_js_topk16_Qwen3-1.7B-Base-OPD_Qwen3-4B-Base-GRPO/summary_topk16.json"
PLOT_DIR     = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/js_divergence/amc23_js_topk16_Qwen3-1.7B-Base-OPD_Qwen3-4B-Base-GRPO/"

APPLY_CHAT_TEMPLATE = True
ENABLE_THINKING     = False
MAX_RESPONSE_TOKENS = 15360

# Fixed top-k truncation (replaces the previous top-p strategy).
TOP_K = 16

# Number of token positions processed per chunk.
# Peak chunk memory ≈ POSITION_CHUNK × vocab_size × 4 bytes (fp32 probs_full × 2 models)
# = 64 × 151936 × 8 ≈ 78 MB — safe.
POSITION_CHUNK = 64

TORCH_DTYPE        = "bfloat16"
NUM_GPUS           = 8
BATCH_TOKEN_BUDGET = 16384   # max (max_seq_len × batch_size) per forward pass

# ============================================================
# Implementation — no need to edit below
# ============================================================

import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def build_sequences(records, tokenizer):
    seqs, skipped = [], 0
    for rec in records:
        if APPLY_CHAT_TEMPLATE:
            formatted  = tokenizer.apply_chat_template(
                [{"role": "user", "content": rec["prompt"]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
            )
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        else:
            prompt_ids = tokenizer.encode(rec["prompt"], add_special_tokens=True)

        response_ids = tokenizer.encode(rec["response"], add_special_tokens=False)
        if MAX_RESPONSE_TOKENS:
            response_ids = response_ids[:MAX_RESPONSE_TOKENS]
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


# ------------------------------------------------------------------ #
# Per-chunk JS computation on four support sets
# ------------------------------------------------------------------ #

def _js_on_support(probs_full_A, probs_full_B, support_mask):
    """
    Compute JS divergence on a per-position support mask.

    probs_full_A, probs_full_B : (C, V) float32 — full-vocab softmax probs
    support_mask               : (C, V) bool    — which tokens belong to support

    Returns: js (C,) float32; positions with empty support → NaN.
    """
    P_A = probs_full_A * support_mask                                   # (C, V)
    P_B = probs_full_B * support_mask                                   # (C, V)

    sum_A = P_A.sum(dim=-1, keepdim=True)                               # (C, 1)
    sum_B = P_B.sum(dim=-1, keepdim=True)                               # (C, 1)

    # Renormalise on the support (equivalent to softmax over support).
    P_A = P_A / sum_A.clamp(min=1e-40)
    P_B = P_B / sum_B.clamp(min=1e-40)

    M = 0.5 * (P_A + P_B)

    kl_A = torch.where(P_A > 0, P_A * torch.log(P_A / M.clamp(min=1e-40)),
                       torch.zeros_like(P_A)).sum(dim=-1)
    kl_B = torch.where(P_B > 0, P_B * torch.log(P_B / M.clamp(min=1e-40)),
                       torch.zeros_like(P_B)).sum(dim=-1)
    js = 0.5 * kl_A + 0.5 * kl_B                                        # (C,)

    # Empty support → NaN
    support_size = support_mask.sum(dim=-1)                             # (C,)
    js = torch.where(support_size > 0, js, torch.full_like(js, float("nan")))
    return js


def compute_js_chunk(logits_A, logits_B, top_k):
    """
    logits_A, logits_B : (C, V) float32 on the same device.

    Returns dict of (C,) float32 tensors:
      js_union, js_intersection, js_student, js_teacher, iou, intersect_size
    """
    C, V = logits_A.shape
    K = min(top_k, V)

    # Full-vocab softmax once per model — used for renormalisation under any support.
    # (Equivalent to softmax-inside-support for any support set ⊆ vocab.)
    probs_full_A = torch.softmax(logits_A, dim=-1)                      # (C, V)
    probs_full_B = torch.softmax(logits_B, dim=-1)                      # (C, V)

    # Top-k indices per model.
    _, topk_idx_A = torch.topk(logits_A, k=K, dim=-1)                   # (C, K)
    _, topk_idx_B = torch.topk(logits_B, k=K, dim=-1)                   # (C, K)

    # Build full-vocab boolean masks for each model's top-k set.
    mask_A = torch.zeros(C, V, dtype=torch.bool, device=logits_A.device)
    mask_A.scatter_(1, topk_idx_A, True)
    mask_B = torch.zeros(C, V, dtype=torch.bool, device=logits_B.device)
    mask_B.scatter_(1, topk_idx_B, True)

    union     = mask_A | mask_B
    intersect = mask_A & mask_B

    js_union        = _js_on_support(probs_full_A, probs_full_B, union)
    js_intersection = _js_on_support(probs_full_A, probs_full_B, intersect)
    js_student      = _js_on_support(probs_full_A, probs_full_B, mask_A)
    js_teacher      = _js_on_support(probs_full_A, probs_full_B, mask_B)

    inter_size = intersect.sum(dim=-1).float()                          # (C,)
    union_size = union.sum(dim=-1).float()                              # (C,)
    iou        = inter_size / union_size.clamp(min=1)                   # (C,)

    return {
        "js_union":        js_union,
        "js_intersection": js_intersection,
        "js_student":      js_student,
        "js_teacher":      js_teacher,
        "iou":             iou,
        "intersect_size":  inter_size,
    }


# ------------------------------------------------------------------ #
# Per-GPU worker
# ------------------------------------------------------------------ #

def score_on_device(rank, seqs_chunk, pad_token_id, tmp_path):
    device = f"cuda:{rank}"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[TORCH_DTYPE]

    print(f"[GPU {rank}] Loading Model A (student): {MODEL_PATH_A}", flush=True)
    model_A = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH_A, dtype=torch_dtype, device_map=device,
        attn_implementation="flash_attention_2", local_files_only=True,
    ).eval()

    print(f"[GPU {rank}] Loading Model B (teacher): {MODEL_PATH_B}", flush=True)
    model_B = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH_B, dtype=torch_dtype, device_map=device,
        attn_implementation="flash_attention_2", local_files_only=True,
    ).eval()

    batches  = make_batches(seqs_chunk, BATCH_TOKEN_BUDGET)
    n_batches = len(batches)
    total_seqs = sum(len(b) for b in batches)
    total_resp_tokens = sum(len(s["response_token_ids"]) for b in batches for s in b)

    print(f"[GPU {rank}] Ready: {n_batches} batches, {total_seqs} sequences, "
          f"{total_resp_tokens} response tokens to score.", flush=True)

    # Only rank 0 shows a tqdm progress bar (single in-place line) so the
    # output of multiple workers does not interleave / flood the screen.
    # Other ranks process silently; the main process prints their exit info
    # via .join().
    show_bar = (rank == 0) and _HAS_TQDM
    pbar = None
    if show_bar:
        pbar = tqdm(
            total=total_seqs,
            desc=f"GPU 0 (of {NUM_GPUS}, representative)",
            unit="seq",
            dynamic_ncols=True,
            mininterval=1.0,
            smoothing=0.1,
        )

    t_start = time.time()
    done_seqs = 0
    done_resp_tokens = 0

    with open(tmp_path, "w", encoding="utf-8") as out_f:
        for batch_idx, batch in enumerate(batches):
            max_len = max(len(s["full_ids"]) for s in batch)

            input_ids = torch.full((len(batch), max_len), pad_token_id,
                                   dtype=torch.long, device=device)
            attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
            for i, s in enumerate(batch):
                ids   = s["full_ids"]
                start = max_len - len(ids)
                input_ids[i, start:] = torch.tensor(ids, dtype=torch.long, device=device)
                attn_mask[i, start:] = 1

            with torch.no_grad():
                logits_all_A = model_A(input_ids=input_ids, attention_mask=attn_mask).logits
                logits_all_B = model_B(input_ids=input_ids, attention_mask=attn_mask).logits

            del input_ids, attn_mask

            for i, s in enumerate(batch):
                offset = max_len - len(s["full_ids"])
                p_len  = s["prompt_len"]
                r_len  = len(s["response_token_ids"])

                resp_A = logits_all_A[i, offset + p_len : offset + p_len + r_len].float()
                resp_B = logits_all_B[i, offset + p_len : offset + p_len + r_len].float()

                js_u, js_i, js_s, js_t, iou_l = [], [], [], [], []
                empty_intersect_count = 0

                for cs in range(0, r_len, POSITION_CHUNK):
                    chunk_A = resp_A[cs : cs + POSITION_CHUNK]
                    chunk_B = resp_B[cs : cs + POSITION_CHUNK]

                    out = compute_js_chunk(chunk_A, chunk_B, TOP_K)

                    js_u.extend(out["js_union"].cpu().tolist())
                    js_i.extend(out["js_intersection"].cpu().tolist())
                    js_s.extend(out["js_student"].cpu().tolist())
                    js_t.extend(out["js_teacher"].cpu().tolist())
                    iou_l.extend(out["iou"].cpu().tolist())
                    empty_intersect_count += int((out["intersect_size"] == 0).sum().item())

                    del chunk_A, chunk_B, out

                del resp_A, resp_B

                def _mean(xs):
                    valid = [v for v in xs if not math.isnan(v)]
                    return sum(valid) / len(valid) if valid else float("nan")

                empty_ratio = empty_intersect_count / r_len if r_len > 0 else 0.0

                out_f.write(json.dumps({
                    "example_id":              s["example_id"],
                    "seed":                    s["seed"],
                    "response_token_ids":      s["response_token_ids"],
                    "js_union":                js_u,
                    "mean_js_union":           _mean(js_u),
                    "js_intersection":         js_i,
                    "mean_js_intersection":    _mean(js_i),
                    "js_student":              js_s,
                    "mean_js_student":         _mean(js_s),
                    "js_teacher":              js_t,
                    "mean_js_teacher":         _mean(js_t),
                    "iou":                     iou_l,
                    "mean_iou":                _mean(iou_l),
                    "empty_intersection_ratio": empty_ratio,
                    "response_length":         r_len,
                }, ensure_ascii=False) + "\n")

                done_seqs += 1
                done_resp_tokens += r_len
                if pbar is not None:
                    pbar.update(1)

            out_f.flush()

            del logits_all_A, logits_all_B
            torch.cuda.empty_cache()

            if pbar is not None:
                elapsed  = time.time() - t_start
                tok_rate = done_resp_tokens / elapsed if elapsed > 0 else 0.0
                pbar.set_postfix(
                    batch=f"{batch_idx + 1}/{n_batches}",
                    tok_per_s=f"{tok_rate:.0f}",
                    tok_done=f"{done_resp_tokens}/{total_resp_tokens}",
                    refresh=False,
                )

    if pbar is not None:
        pbar.close()

    total_elapsed = time.time() - t_start
    print(f"[GPU {rank}] Finished {done_seqs} sequences "
          f"({done_resp_tokens} response tokens) in {total_elapsed / 60:.1f} min.",
          flush=True)


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #

def plot_distribution(values, xlabel, title, out_path, bins=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(values)
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        print(f"[plot] No valid values for {title}; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    counts, edges = np.histogram(arr, bins=bins)
    proportions    = counts / counts.sum()
    ax.bar(edges[:-1], proportions, width=np.diff(edges), align="edge",
           color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Proportion (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)

    # Log y-axis: 10^0, 10^-1, 10^-2, ... Bottom is the smallest non-zero
    # proportion (one position out of n), top is 10^0. Empty bins are not
    # drawn on a log axis (proportion=0 is invalid in log space).
    ax.set_yscale("log")
    nonzero = proportions[proportions > 0]
    y_min = nonzero.min() if nonzero.size else 1.0 / max(arr.size, 1)
    # Snap lower bound down to the next power of 10 so ticks align cleanly.
    y_low = 10 ** math.floor(math.log10(y_min))
    ax.set_ylim(y_low, 1.0)

    ax.text(0.98, 0.95,
            f"n={arr.size}\nmean={arr.mean():.4f}\nmedian={np.median(arr):.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    import numpy as np

    print(f"[Config] Model A (student) = {MODEL_PATH_A}")
    print(f"[Config] Model B (teacher) = {MODEL_PATH_B}")
    print(f"[Config] rollout           = {ROLLOUT_PATH}")
    print(f"[Config] output            = {OUTPUT_PATH}")
    print(f"[Config] top_k             = {TOP_K}")
    print(f"[Config] num_gpus          = {NUM_GPUS}  |  chunk = {POSITION_CHUNK}")

    # --- Tokenizer (use Model A's tokenizer; must match the one used during generation) ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_A, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Load rollout records ---
    print("Loading rollout records...")
    records = []
    with open(ROLLOUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records.")

    seqs = build_sequences(records, tokenizer)

    # --- GPU allocation ---
    available = torch.cuda.device_count()
    n_gpus    = min(NUM_GPUS, available)
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs available.")
    print(f"\n[GPU] Using {n_gpus}/{available} GPUs.")

    chunks   = [seqs[i::n_gpus] for i in range(n_gpus)]
    out_dir  = Path(OUTPUT_PATH).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_paths = [out_dir / f"_js_topk_tmp_gpu{r}.jsonl" for r in range(n_gpus)]

    for r in range(n_gpus):
        print(f"[GPU {r}] assigned {len(chunks[r])} sequences "
              f"({sum(len(s['response_token_ids']) for s in chunks[r])} response tokens).")

    t_main = time.time()
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(n_gpus):
        if not chunks[rank]:
            continue
        p = ctx.Process(
            target=score_on_device,
            args=(rank, chunks[rank], tokenizer.pad_token_id, tmp_paths[rank]),
        )
        p.start()
        procs.append((rank, p))

    for rank, p in procs:
        p.join()
        print(f"[Main] GPU {rank} worker exited with code {p.exitcode} "
              f"(elapsed {(time.time() - t_main) / 60:.1f} min).", flush=True)

    failed = [(r, p) for r, p in procs if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} worker(s) failed: " +
                           ", ".join(f"GPU {r} code={p.exitcode}" for r, p in failed))

    # --- Merge shards ---
    print("\nMerging shards...")
    all_results = []
    shard_iter = range(n_gpus)
    if _HAS_TQDM:
        shard_iter = tqdm(shard_iter, desc="merging shards", unit="shard")
    for rank in shard_iter:
        path = tmp_paths[rank]
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))
        path.unlink()

    all_results.sort(key=lambda r: (r["example_id"], r["seed"]))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_results)} records to {OUTPUT_PATH}.")

    # --- Global statistics ---
    def _flat_valid(key):
        return [v for rec in all_results for v in rec[key] if not math.isnan(v)]

    all_js_u = _flat_valid("js_union")
    all_js_i = _flat_valid("js_intersection")   # NaN positions (empty intersection) already filtered
    all_js_s = _flat_valid("js_student")
    all_js_t = _flat_valid("js_teacher")
    all_iou  = _flat_valid("iou")

    total_positions       = sum(rec["response_length"] for rec in all_results)
    total_empty_intersect = sum(
        rec["empty_intersection_ratio"] * rec["response_length"] for rec in all_results
    )
    global_empty_ratio = total_empty_intersect / total_positions if total_positions > 0 else 0.0

    def _stats(arr, name):
        a = np.array(arr)
        return {
            f"global_mean_{name}":   float(a.mean()) if a.size else float("nan"),
            f"global_median_{name}": float(np.median(a)) if a.size else float("nan"),
            f"global_p95_{name}":    float(np.percentile(a, 95)) if a.size else float("nan"),
            f"global_p99_{name}":    float(np.percentile(a, 99)) if a.size else float("nan"),
        }

    summary = {
        "total_records":      len(all_results),
        "total_positions":    total_positions,
        "model_a_student":    MODEL_PATH_A,
        "model_b_teacher":    MODEL_PATH_B,
        "top_k":              TOP_K,
        "empty_intersection_ratio": global_empty_ratio,
        "n_valid_js_intersection":  len(all_js_i),
        **_stats(all_js_u, "js_union"),
        **_stats(all_js_i, "js_intersection"),
        **_stats(all_js_s, "js_student"),
        **_stats(all_js_t, "js_teacher"),
        "global_mean_iou":    float(np.mean(all_iou)) if all_iou else float("nan"),
        "global_median_iou":  float(np.median(all_iou)) if all_iou else float("nan"),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {SUMMARY_PATH}.")

    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:32s} = {v:.6f}")
        else:
            print(f"  {k:32s} = {v}")

    # --- Plots ---
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    name_a = Path(MODEL_PATH_A).name
    name_b = Path(MODEL_PATH_B).name
    suffix = f"top-k={TOP_K}"

    plot_distribution(
        all_js_u,
        xlabel=f"JS Divergence (nats, support=union, {suffix})",
        title=f"Per-token JS — UNION support\n{name_a} (student) vs {name_b} (teacher)",
        out_path=Path(PLOT_DIR) / "js_distribution_union.png",
    )
    plot_distribution(
        all_js_i,
        xlabel=f"JS Divergence (nats, support=intersection, {suffix})",
        title=f"Per-token JS — INTERSECTION support (empty-intersection skipped)\n{name_a} vs {name_b}",
        out_path=Path(PLOT_DIR) / "js_distribution_intersection.png",
    )
    plot_distribution(
        all_js_s,
        xlabel=f"JS Divergence (nats, support=student top-k, {suffix})",
        title=f"Per-token JS — STUDENT top-k support\n{name_a} (student) vs {name_b} (teacher)",
        out_path=Path(PLOT_DIR) / "js_distribution_student.png",
    )
    plot_distribution(
        all_js_t,
        xlabel=f"JS Divergence (nats, support=teacher top-k, {suffix})",
        title=f"Per-token JS — TEACHER top-k support\n{name_a} (student) vs {name_b} (teacher)",
        out_path=Path(PLOT_DIR) / "js_distribution_teacher.png",
    )
    plot_distribution(
        all_iou,
        xlabel=f"IoU of top-{TOP_K} token sets",
        title=f"Per-token Top-k Set IoU Distribution\n{name_a} vs {name_b}",
        out_path=Path(PLOT_DIR) / "iou_distribution.png",
    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
