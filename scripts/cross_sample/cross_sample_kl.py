"""
Cross-sampling generation with student/teacher routing on KL divergence.

Two routing directions are supported (env var KL_DIRECTION):
  rkl : KL(student || teacher)   — reverse KL, mass-seeking on the student
  fkl : KL(teacher || student)   — forward KL, mass-seeking on the teacher

For each token position:
  1. student and teacher each produce next-token logits
  2. decide_sampler(...) — picks "student" or "teacher" based on a KL check
     on the UNION of the two models' top-k token sets (computed on T=1 logits)
  3. the chosen model's logits are sampled at TEMPERATURE to produce the next
     token, which is appended to BOTH models' contexts (KV-cache reused)
  4. loop until EOS / stop-token / MAX_TOKENS

Then (unless --skip-grade) the responses are graded with grade_answer_verl
and avg@N is reported.

Use --skip-gen --grade-file <path> to grade an existing rollout file without
generating.
"""

# ============================================================================ #
#                  Global parameters — edit here before running               #
# ============================================================================ #

import os
import sys
import gc
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make grade_answer_verl importable.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "val" / "eval"))
from utils import grade_answer_verl  # noqa: E402

# --- Models ---
STUDENT_MODEL = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-1.7B-Base"
TEACHER_MODEL = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-4B-Base-GRPO"

# --- Routing (cross-sampling) ---
# Overridable via env vars KL_THRESHOLD / KL_DIRECTION
# (e.g. `KL_DIRECTION=rkl KL_THRESHOLD=0.12 python cross_sample_kl.py`).
# Falls back to the defaults below when unset, so the script still runs directly.
KL_THRESHOLD = float(os.environ.get("KL_THRESHOLD", "0.12"))  # KL > threshold → teacher; else student
KL_DIRECTION = os.environ.get("KL_DIRECTION", "rkl").lower()
if KL_DIRECTION not in {"rkl", "fkl"}:
    raise ValueError("KL_DIRECTION must be 'rkl' (KL(student||teacher)) or 'fkl' (KL(teacher||student))")
ROUTER_TOP_K = 16            # support = union of each model's top-k (T=1)
KL_TEMPERATURE = 1.0         # KL computed on T=KL_TEMPERATURE distributions

# --- Sampling ---
TEMPERATURE = 0.7
MAX_TOKENS = 7168
# no top-p, no top-k truncation on sampling — pure temperature sampling

# --- Batching / parallelism ---
BATCH_SIZE = 16             # number of sequences cross-sampled in parallel per GPU

# --- Data ---
DATA_DIR = "/mmu_cd_ssd/pengtiantian/projects/OPD/scripts/val/data"
TASKS = [
    {"name": "AMC23", "path": f"{DATA_DIR}/AMC23/test.parquet", "N": 8},
]
ENABLE_THINKING = False

# --- Execution ---
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
TORCH_DTYPE = "bfloat16"

# --- Output ---
OUT_ROOT = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample"
PROMPT_TEMPLATE = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."

REPLACE = False   # overwrite existing rollout file
APPEND = True     # load existing rollouts and only generate missing (sample, seed) pairs


def kl_direction_label(kl_direction: str = KL_DIRECTION) -> str:
    """Human-readable form of the configured KL direction."""
    return ("KL(student||teacher)" if kl_direction == "rkl"
            else "KL(teacher||student)")


def make_run_tag() -> str:
    """Tag identifying this routing run (used in output dir name)."""
    stu = Path(STUDENT_MODEL).name
    tch = Path(TEACHER_MODEL).name
    return f"cross_sample_{stu}_TCH_{tch}_{KL_DIRECTION}th{KL_THRESHOLD}_topk{ROUTER_TOP_K}"


def make_rollout_filename(task_name: str, n: int) -> str:
    return f"{task_name.lower()}_t{TEMPERATURE}_n{n}-MNT{MAX_TOKENS}.jsonl"


# ============================================================================ #
#                              Routing helpers                                 #
# ============================================================================ #

def kl_divergence_on_union_batched(
    logits_stu: torch.Tensor,
    logits_tch: torch.Tensor,
    top_k: int = ROUTER_TOP_K,
    kl_temperature: float = KL_TEMPERATURE,
    kl_direction: str = KL_DIRECTION,
) -> torch.Tensor:
    """
    Batched KL divergence on UNION top-k support.

    Args:
        logits_stu, logits_tch: (B, V) float tensors on the same device.
        kl_direction: "rkl" → KL(student||teacher); "fkl" → KL(teacher||student).

    Returns:
        kl: (B,) float32 tensor on the same device. Per row, the union support
            can have different sizes (≤ 2K); we build a (B, 2K) gather of
            *unique-per-row* indices and mask duplicates so they don't double-
            count in the softmax denominator.
    """
    z_s = logits_stu.float() / kl_temperature
    z_t = logits_tch.float() / kl_temperature
    B, V = z_s.shape

    K = min(top_k, V)
    topk_s = torch.topk(z_s, k=K, dim=-1).indices                # (B, K)
    topk_t = torch.topk(z_t, k=K, dim=-1).indices                # (B, K)
    cat_idx = torch.cat([topk_s, topk_t], dim=-1)                # (B, 2K)

    # Per-row "first occurrence" mask: keep entry j iff cat_idx[b, j] doesn't
    # appear earlier in row b. Equivalent to torch.unique per-row but vectorised.
    # (B, 2K, 2K): eq[b, i, j] = (cat_idx[b, i] == cat_idx[b, j])
    eq = cat_idx.unsqueeze(-1) == cat_idx.unsqueeze(-2)
    # Lower triangle (j < i) gives "earlier occurrence" of value at position i.
    earlier = torch.tril(eq, diagonal=-1)                        # (B, 2K, 2K)
    duplicate = earlier.any(dim=-1)                              # (B, 2K)
    keep_mask = ~duplicate                                       # (B, 2K)

    # Gather logits at union positions (duplicates included, masked below).
    z_s_u = torch.gather(z_s, 1, cat_idx)                        # (B, 2K)
    z_t_u = torch.gather(z_t, 1, cat_idx)                        # (B, 2K)

    # Masked softmax on the per-row unique-union support.
    NEG_INF = torch.finfo(z_s_u.dtype).min
    z_s_u = torch.where(keep_mask, z_s_u, torch.full_like(z_s_u, NEG_INF))
    z_t_u = torch.where(keep_mask, z_t_u, torch.full_like(z_t_u, NEG_INF))

    p_s = torch.softmax(z_s_u, dim=-1)                           # (B, 2K)
    p_t = torch.softmax(z_t_u, dim=-1)                           # (B, 2K)

    eps = 1e-40
    log_ps = torch.log(p_s.clamp(min=eps))
    log_pt = torch.log(p_t.clamp(min=eps))

    # Masked positions have p_s = p_t = 0 so contributions are 0 automatically.
    if kl_direction == "rkl":
        # KL(student || teacher) = sum_x p_s(x) * (log p_s(x) - log p_t(x))
        return (p_s * (log_ps - log_pt)).sum(dim=-1)             # (B,)
    # KL(teacher || student) = sum_x p_t(x) * (log p_t(x) - log p_s(x))
    return (p_t * (log_pt - log_ps)).sum(dim=-1)                 # (B,)


def decide_sampler_batched(
    logits_stu: torch.Tensor,
    logits_tch: torch.Tensor,
    *,
    top_k: int = ROUTER_TOP_K,
    kl_threshold: float = KL_THRESHOLD,
    kl_temperature: float = KL_TEMPERATURE,
    kl_direction: str = KL_DIRECTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched routing decision.

    Args:
        logits_stu, logits_tch: (B, V) on same device.

    Returns:
        use_teacher: (B,) bool tensor on same device — True for teacher route.
        kl_values:   (B,) float32 tensor — per-row KL values in nats.
    """
    kl = kl_divergence_on_union_batched(logits_stu, logits_tch, top_k=top_k,
                                        kl_temperature=kl_temperature,
                                        kl_direction=kl_direction)
    use_teacher = kl > kl_threshold
    return use_teacher, kl


# ============================================================================ #
#                       Per-sequence cross-sampling                            #
# ============================================================================ #

def _resolve_stop_token_ids(tokenizer) -> set[int]:
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            enc = tokenizer.encode(tok, add_special_tokens=False)
            if enc:
                stop_ids.add(int(enc[0]))
        except Exception:
            pass
    return stop_ids


def cross_sample_batch(
    student,
    teacher,
    tokenizer,
    prompt_ids_list: list[list[int]],
    *,
    max_new_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    stop_token_ids: set[int] | None = None,
    generator: torch.Generator | None = None,
) -> list[dict]:
    """
    Batched per-token cross-sampling. Generates len(prompt_ids_list) responses
    in parallel on a single GPU.

    Strategy:
      - prompts are LEFT-padded to a common length for prefill
      - student/teacher each forward on a separate CUDA stream so the smaller
        student forward overlaps with the teacher forward
      - per-token state (response ids, router decisions, KL values) is kept in
        preallocated GPU tensors of shape (B, max_new_tokens) — no per-step
        CPU sync
      - early-exit check `done.all()` runs every DONE_CHECK_EVERY steps
      - sampling uses a single batched multinomial driven by `generator`
        (per-batch deterministic seeding gives reproducibility for the same
        batching order; bit-exact per-(sample, seed) reproducibility was
        dropped to remove a per-row kernel-launch loop)

    Returns: list of per-sequence result dicts.
    """
    B = len(prompt_ids_list)
    assert B >= 1

    device = next(student.parameters()).device
    stop_token_ids = stop_token_ids if stop_token_ids is not None else set()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    assert pad_id is not None

    stop_ids_tensor = torch.tensor(sorted(stop_token_ids), dtype=torch.long,
                                   device=device) if stop_token_ids else None

    # --- Build LEFT-padded prefill batch ---
    max_plen = max(len(p) for p in prompt_ids_list)
    input_ids = torch.full((B, max_plen), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, max_plen), dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids_list):
        start = max_plen - len(p)
        input_ids[i, start:] = torch.tensor(p, dtype=torch.long, device=device)
        attn_mask[i, start:] = 1

    stream_s = torch.cuda.Stream(device=device)
    stream_t = torch.cuda.Stream(device=device)

    def _two_forwards(ids_in, attn_in, past_s, past_t):
        """Run student + teacher forwards on separate streams."""
        default_stream = torch.cuda.current_stream(device=device)
        stream_s.wait_stream(default_stream)
        stream_t.wait_stream(default_stream)
        with torch.cuda.stream(stream_s):
            out_s = student(input_ids=ids_in, attention_mask=attn_in,
                            past_key_values=past_s, use_cache=True)
        with torch.cuda.stream(stream_t):
            out_t = teacher(input_ids=ids_in, attention_mask=attn_in,
                            past_key_values=past_t, use_cache=True)
        default_stream.wait_stream(stream_s)
        default_stream.wait_stream(stream_t)
        return out_s, out_t

    DONE_CHECK_EVERY = 32

    with torch.inference_mode():
        # --- Prefill ---
        out_s, out_t = _two_forwards(input_ids, attn_mask, None, None)
        past_s = out_s.past_key_values
        past_t = out_t.past_key_values
        last_logits_s = out_s.logits[:, -1, :]                       # (B, V)
        last_logits_t = out_t.logits[:, -1, :]                       # (B, V)
        del out_s, out_t

        # --- Preallocated GPU-side per-token state ---
        resp_ids_buf  = torch.full((B, max_new_tokens), pad_id,
                                   dtype=torch.long, device=device)
        decisions_buf = torch.zeros((B, max_new_tokens),
                                    dtype=torch.bool, device=device)
        kl_buf        = torch.zeros((B, max_new_tokens),
                                    dtype=torch.float32, device=device)
        lengths       = torch.full((B,), max_new_tokens,
                                   dtype=torch.long, device=device)
        done          = torch.zeros(B, dtype=torch.bool, device=device)
        finish_stop   = torch.zeros(B, dtype=torch.bool, device=device)
        cur_attn = attn_mask                                         # (B, plen) so far

        for step in range(max_new_tokens):
            # Routing.
            use_teacher, kl_b = decide_sampler_batched(last_logits_s, last_logits_t)
            logits_pick = torch.where(use_teacher.unsqueeze(-1),
                                      last_logits_t, last_logits_s)  # (B, V)

            # Sample next tokens (single batched multinomial).
            if temperature <= 0.0:
                next_ids = torch.argmax(logits_pick, dim=-1)         # (B,)
            else:
                probs = torch.softmax(logits_pick.float() / temperature, dim=-1)
                next_ids = torch.multinomial(
                    probs, num_samples=1, generator=generator
                ).squeeze(-1)

            # Already-done rows → PAD (their model outputs are ignored anyway).
            next_ids = torch.where(done, torch.full_like(next_ids, pad_id), next_ids)

            # Record per-token state — pure GPU writes, no CPU sync.
            resp_ids_buf[:, step]  = next_ids
            decisions_buf[:, step] = use_teacher
            kl_buf[:, step]        = kl_b

            # Stop detection (vectorised).
            if stop_ids_tensor is not None:
                stop_hit = (next_ids.unsqueeze(-1) == stop_ids_tensor).any(-1)
                new_done = stop_hit & ~done
                lengths = torch.where(new_done,
                                      torch.full_like(lengths, step + 1), lengths)
                finish_stop = finish_stop | new_done
                done = done | stop_hit

            # Amortised early-exit check (1 sync per DONE_CHECK_EVERY steps).
            if (step + 1) % DONE_CHECK_EVERY == 0 and bool(done.all().item()):
                break
            if step + 1 == max_new_tokens:
                break  # no need to do another forward

            # Step forward both models on the new tokens.
            next_input = next_ids.unsqueeze(-1)                      # (B, 1)
            attn_new_col = (~done).long().unsqueeze(-1)              # (B, 1)
            cur_attn = torch.cat([cur_attn, attn_new_col], dim=-1)

            out_s, out_t = _two_forwards(next_input, cur_attn, past_s, past_t)
            past_s = out_s.past_key_values
            past_t = out_t.past_key_values
            last_logits_s = out_s.logits[:, -1, :]
            last_logits_t = out_t.logits[:, -1, :]
            del out_s, out_t

    # Free.
    del past_s, past_t, last_logits_s, last_logits_t

    # Single CPU sync at end.
    lengths_cpu     = lengths.tolist()
    finish_stop_cpu = finish_stop.tolist()
    resp_ids_cpu    = resp_ids_buf.tolist()
    decisions_cpu   = decisions_buf.tolist()
    kl_cpu          = kl_buf.tolist()

    results = []
    for i in range(B):
        L = lengths_cpu[i]
        results.append({
            "response_token_ids": resp_ids_cpu[i][:L],
            "router_decisions":   ["teacher" if d else "student"
                                   for d in decisions_cpu[i][:L]],
            "kl_values":          kl_cpu[i][:L],
            "finish_reason":      "stop" if finish_stop_cpu[i] else "length",
        })
    return results


# ============================================================================ #
#                        Data loading / sharding                               #
# ============================================================================ #

def load_samples(filepath: str) -> list[dict]:
    df = pd.read_parquet(filepath)
    if any(tag in filepath for tag in ("BRUMO25", "CMIMC25", "HMMT25")):
        samples = [
            {"example_id": i,
             "prompt": df.at[i, "problem"].strip(),
             "answer": df.at[i, "answer"].strip()}
            for i in range(len(df))
        ]
    else:
        samples = [
            {"example_id": i,
             "prompt": df.at[i, "prompt"][0]["content"].strip(),
             "answer": df.at[i, "reward_model"]["ground_truth"].strip()}
            for i in range(len(df))
        ]
    print(f"Loaded {len(samples)} samples from {filepath}")
    return samples


def split_units(units: list[tuple], num_workers: int) -> list[list[tuple]]:
    """Round-robin split of (sample_idx, seed) work-units across workers."""
    chunks = [[] for _ in range(num_workers)]
    for i, u in enumerate(units):
        chunks[i % num_workers].append(u)
    return chunks


# ============================================================================ #
#                              Worker process                                  #
# ============================================================================ #

def worker_process(args_tuple):
    """
    args_tuple = (rank, gpu_id, work_units, samples, tmp_path, enable_thinking)
        work_units : list[(sample_idx, seed)]
        samples    : list[dict] (the full sample list; indexed by sample_idx)
        tmp_path   : path to per-worker output jsonl
    """
    rank, gpu_id, work_units, samples, tmp_path, enable_thinking = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[TORCH_DTYPE]
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES already pinned to gpu_id

    try:
        print(f"[GPU {gpu_id}] rank={rank} | {len(work_units)} units | "
              f"loading student + teacher...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        student = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL, dtype=torch_dtype, device_map=device,
            attn_implementation="flash_attention_2", local_files_only=True,
        ).eval()
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL, dtype=torch_dtype, device_map=device,
            attn_implementation="flash_attention_2", local_files_only=True,
        ).eval()

        stop_token_ids = _resolve_stop_token_ids(tokenizer)
        print(f"[GPU {gpu_id}] models ready | stop_token_ids={sorted(stop_token_ids)}", flush=True)

        # Per-batch generator: deterministic seed derived from the batch contents
        # so the same batching order reproduces. (Strict per-(sample,seed)
        # reproducibility was traded away for one batched multinomial per step
        # instead of B serial calls.)
        def _gen_for_batch(batch_units: list[tuple[int, int]]) -> torch.Generator:
            seed = 0
            for si, sd in batch_units:
                seed = (seed * 1_000_003 + (si + 1) * 1_000_003 + sd) & 0x7FFFFFFF
            g = torch.Generator(device=device)
            g.manual_seed(seed if seed != 0 else 1)
            return g

        # Pre-format prompts once.
        formatted_cache: dict[int, list[int]] = {}
        def _prompt_ids_for(sample_idx: int) -> list[int]:
            if sample_idx in formatted_cache:
                return formatted_cache[sample_idx]
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": samples[sample_idx]["prompt"]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            ids = tokenizer.encode(formatted, add_special_tokens=False)
            formatted_cache[sample_idx] = ids
            return ids

        show_bar = (rank == 0)
        pbar = tqdm(
            total=len(work_units), desc=f"GPU {gpu_id} (of {len(GPUS)}, representative)",
            unit="seq", dynamic_ncols=True, mininterval=1.0, smoothing=0.1,
            disable=not show_bar,
        )

        t_start = time.time()
        done_tokens = 0
        n_batches_total = (len(work_units) + BATCH_SIZE - 1) // BATCH_SIZE

        with open(tmp_path, "w", encoding="utf-8") as fout:
            # Process work_units in mini-batches of BATCH_SIZE.
            for batch_start in range(0, len(work_units), BATCH_SIZE):
                cur_batch_idx = batch_start // BATCH_SIZE + 1
                batch_units = work_units[batch_start:batch_start + BATCH_SIZE]
                prompt_ids_list = [_prompt_ids_for(si) for si, _ in batch_units]
                gen = _gen_for_batch(batch_units)

                try:
                    results = cross_sample_batch(
                        student, teacher, tokenizer, prompt_ids_list,
                        max_new_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        stop_token_ids=stop_token_ids,
                        generator=gen,
                    )
                except Exception as e:
                    print(f"[GPU {gpu_id}] batch starting at unit {batch_start} FAILED: {e}", flush=True)
                    import traceback; traceback.print_exc()
                    torch.cuda.empty_cache()
                    continue

                for (sample_idx, seed), result in zip(batch_units, results):
                    sample = samples[sample_idx]
                    resp_text = tokenizer.decode(result["response_token_ids"], skip_special_tokens=True)
                    n_tch = sum(1 for d in result["router_decisions"] if d == "teacher")
                    n_total = len(result["router_decisions"])
                    teacher_ratio = (n_tch / n_total) if n_total > 0 else 0.0
                    mean_kl = (sum(result["kl_values"]) / n_total) if n_total > 0 else 0.0

                    fout.write(json.dumps({
                        "example_id": sample["example_id"],
                        "prompt": sample["prompt"],
                        "answer": sample["answer"],
                        "seed": seed,
                        "response": resp_text,
                        "response_token_ids": result["response_token_ids"],
                        "router_decisions": result["router_decisions"],
                        "kl_values": result["kl_values"],
                        "kl_direction": KL_DIRECTION,
                        "teacher_ratio": teacher_ratio,
                        "mean_kl": mean_kl,
                        "response_length": n_total,
                        "finish_reason": result["finish_reason"],
                    }, ensure_ascii=False) + "\n")

                    done_tokens += n_total
                    pbar.update(1)

                if show_bar:
                    elapsed = time.time() - t_start
                    tok_rate = done_tokens / elapsed if elapsed > 0 else 0.0
                    pbar.set_postfix(
                        bch=f"{cur_batch_idx}/{n_batches_total}",
                        tok_per_s=f"{tok_rate:.0f}",
                        refresh=True,
                    )

                fout.flush()
                # torch.cuda.empty_cache()

        pbar.close()
        print(f"[GPU {gpu_id}] done {len(work_units)} units, "
              f"{done_tokens} tokens, {(time.time() - t_start) / 60:.1f} min.",
              flush=True)

    except Exception as e:
        print(f"[GPU {gpu_id}] CRITICAL: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        try:
            del student, teacher
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================ #
#                              Generation driver                               #
# ============================================================================ #

def run_generation(enable_thinking: bool):
    print(f"\n{'='*70}")
    print(f"  Cross-sampling generation (KL routing)")
    print(f"  student  = {STUDENT_MODEL}")
    print(f"  teacher  = {TEACHER_MODEL}")
    print(f"  routing  = {KL_DIRECTION.upper()} {kl_direction_label()} > {KL_THRESHOLD}"
          f" (union top-{ROUTER_TOP_K}, T={KL_TEMPERATURE}) → teacher else student")
    print(f"  sampling = T={TEMPERATURE}, max_tokens={MAX_TOKENS}")
    print(f"  batch    = {BATCH_SIZE}/GPU")
    print(f"  GPUs     = {GPUS}")
    print(f"{'='*70}\n")

    run_tag = make_run_tag()
    out_dir = Path(OUT_ROOT) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        task_name = task["name"]
        task_path = task["path"]
        N = task["N"]
        out_path = out_dir / make_rollout_filename(task_name, N)

        print(f"\n--- Task: {task_name} (N={N}) → {out_path}")

        if out_path.exists() and not REPLACE and not APPEND:
            print(f"  exists and REPLACE=APPEND=False → skip.")
            continue

        # Load existing rollouts (APPEND mode).
        existing: list[dict] = []
        done_pairs: set[tuple[int, int]] = set()
        if APPEND and not REPLACE and out_path.exists():
            print(f"  APPEND: reading existing rollouts from {out_path}")
            with out_path.open() as f:
                for line in f:
                    item = json.loads(line)
                    existing.append(item)
                    done_pairs.add((int(item["example_id"]), int(item["seed"])))
            print(f"  APPEND: {len(done_pairs)} (sample,seed) pairs already done.")

        samples = load_samples(task_path)
        for s in samples:
            s["prompt"] = PROMPT_TEMPLATE.format(problem=s["prompt"])
        if samples:
            print(f"  example formatted prompt:\n    {samples[0]['prompt'][:200]}...")

        # Build work-units = (sample_idx, seed) not already done.
        # NOTE: example_id == sample_idx here (load_samples sets it that way).
        work_units: list[tuple[int, int]] = []
        for sample_idx in range(len(samples)):
            for seed in range(N):
                if (sample_idx, seed) not in done_pairs:
                    work_units.append((sample_idx, seed))

        if not work_units:
            print(f"  all {len(samples) * N} units already done; nothing to generate.")
            continue

        print(f"  generating {len(work_units)} units across {len(GPUS)} GPU(s).")

        # Shard work across GPUs.
        chunks = split_units(work_units, len(GPUS))
        tmp_paths = [out_dir / f"_tmp_gpu{r}.jsonl" for r in range(len(GPUS))]

        args_list = [
            (rank, gpu_id, chunks[rank], samples, str(tmp_paths[rank]), enable_thinking)
            for rank, gpu_id in enumerate(GPUS)
            if chunks[rank]
        ]

        ctx = mp.get_context("spawn")
        procs = []
        for tup in args_list:
            p = ctx.Process(target=worker_process, args=(tup,))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                print(f"  WARNING: worker exited with code {p.exitcode}")

        # Merge worker outputs with existing.
        all_results = list(existing)
        for r in range(len(GPUS)):
            tp = tmp_paths[r]
            if not tp.exists():
                continue
            with tp.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_results.append(json.loads(line))
            tp.unlink()

        all_results.sort(key=lambda x: (int(x["example_id"]), int(x["seed"])))
        with out_path.open("w", encoding="utf-8") as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  wrote {len(all_results)} rollouts to {out_path}")


# ============================================================================ #
#                                  Grading                                     #
# ============================================================================ #

def grade_file(file_path: Path) -> dict:
    """Rule-based grading; returns dict with avg@N and per-problem detail."""
    print(f"\n--- Grading: {file_path}")
    # Group by example_id.
    records: dict[int, list[dict]] = {}
    with file_path.open() as f:
        for line in f:
            item = json.loads(line)
            records.setdefault(int(item["example_id"]), []).append(item)

    if not records:
        print("  (empty file)")
        return {}

    per_problem: list[dict] = []
    all_correct_flags: list[bool] = []
    teacher_ratios: list[float] = []
    response_lengths: list[int] = []

    for ex_id in sorted(records.keys()):
        items = records[ex_id]
        gt = items[0]["answer"]
        question = items[0].get("prompt", "")
        flags = []
        for it in items:
            ok = bool(grade_answer_verl(it["response"], gt))
            flags.append(ok)
            all_correct_flags.append(ok)
            teacher_ratios.append(float(it.get("teacher_ratio", 0.0)))
            response_lengths.append(int(it.get("response_length", 0)))
        per_problem.append({
            "example_id": ex_id,
            "n": len(items),
            "correct": int(sum(flags)),
            "avg_score": sum(flags) / len(flags) if flags else 0.0,
            "question": question[:200],
            "answer": gt,
        })

    N = per_problem[0]["n"] if per_problem else 0
    avg_at_n = sum(p["avg_score"] for p in per_problem) / len(per_problem)
    best_at_n = sum(1 for p in per_problem if p["correct"] > 0) / len(per_problem)
    solve_none = sum(1 for p in per_problem if p["correct"] == 0)
    solve_all = sum(1 for p in per_problem if p["correct"] == p["n"])

    summary = {
        "file": str(file_path),
        "student": STUDENT_MODEL,
        "teacher": TEACHER_MODEL,
        "kl_direction": KL_DIRECTION,
        "kl_formula": kl_direction_label(),
        "kl_threshold": KL_THRESHOLD,
        "router_top_k": ROUTER_TOP_K,
        "kl_temperature": KL_TEMPERATURE,
        "sampling_temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "n_problems": len(per_problem),
        "n_per_problem": N,
        f"avg@{N}": avg_at_n,
        f"pass@{N}": best_at_n,
        "solve_none": solve_none,
        "solve_all": solve_all,
        "mean_teacher_ratio": sum(teacher_ratios) / len(teacher_ratios) if teacher_ratios else 0.0,
        "mean_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0.0,
        "per_problem": per_problem,
    }

    print(f"  problems         : {summary['n_problems']}")
    print(f"  N per problem    : {summary['n_per_problem']}")
    print(f"  avg@{N:<3}         : {avg_at_n:.4f}")
    print(f"  pass@{N:<3}        : {best_at_n:.4f}")
    print(f"  solve_none/all   : {solve_none} / {solve_all}")
    print(f"  mean teacher_ratio : {summary['mean_teacher_ratio']:.4f}")
    print(f"  mean response_length: {summary['mean_response_length']:.1f}")

    return summary


def run_grading(grade_file_override: str | None = None):
    if grade_file_override:
        files = [Path(grade_file_override)]
    else:
        run_tag = make_run_tag()
        out_dir = Path(OUT_ROOT) / run_tag
        if not out_dir.exists():
            print(f"Output dir does not exist: {out_dir}")
            return
        files = sorted(out_dir.glob("*.jsonl"))

    all_summaries = []
    for fp in files:
        if not fp.exists():
            print(f"Skip missing: {fp}")
            continue
        summary = grade_file(fp)
        if summary:
            all_summaries.append(summary)
            # Save per-file summary next to the rollout file.
            sum_path = fp.with_name(fp.stem + "_grading.json")
            with sum_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"  summary saved to {sum_path}")

    # Combined summary (no per_problem detail).
    if all_summaries:
        combined = [{k: v for k, v in s.items() if k != "per_problem"} for s in all_summaries]
        combined_path = (Path(grade_file_override).parent if grade_file_override
                         else Path(OUT_ROOT) / make_run_tag()) / "grading_summary.json"
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"\nCombined summary saved to {combined_path}")


# ============================================================================ #
#                                   Main                                       #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation.")
    parser.add_argument("--skip-grade", action="store_true", help="Skip grading.")
    parser.add_argument("--grade-file", default=None,
                        help="Grade only the given rollout .jsonl file (implies --skip-gen).")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking_group.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=ENABLE_THINKING)
    args = parser.parse_args()

    if args.grade_file:
        args.skip_gen = True

    if not args.skip_gen:
        print("\n" + "=" * 70)
        print("  STEP 1 / 2 : Cross-sampling generation (KL routing)")
        print("=" * 70)
        run_generation(enable_thinking=args.enable_thinking)

    if not args.skip_grade:
        print("\n" + "=" * 70)
        print("  STEP 2 / 2 : Grading")
        print("=" * 70)
        run_grading(grade_file_override=args.grade_file)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
