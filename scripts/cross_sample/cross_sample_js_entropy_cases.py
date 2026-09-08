"""
Cross-sampling with a JS gate followed by one top-k entropy case.

At every token position:
  1. compute JS divergence on the union of student/teacher top-k tokens
  2. compute each model's entropy on its own renormalised top-k distribution
  3. use teacher iff JS > JS_THRESHOLD and the configured entropy case matches
  4. append the sampled token to both contexts and repeat the full decision

There is no cooldown after a teacher token. Consecutive teacher tokens are
allowed when consecutive positions satisfy the configured condition.
"""

# ============================================================================ #
#                  Global parameters - keep aligned with cross_sample.py       #
# ============================================================================ #

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import cross_sample as base  # noqa: E402


# --- Models ---
STUDENT_MODEL = "/mmu_cd_ssd/pengtiantian/projects/OPD/checkpoint_rerun0820/1.7B-4B_SPARSE-RKL20%_token_reward_direct_DAPO-Math-17k_Qwen3-1.7B-Base_Qwen3-4B-Base-GRPO_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_union-rw_sparse_rkl-2026-08-20_20-08-24/global_step_279/1.7B-4B_SPARSE-RKL20%_step279"
TEACHER_MODEL = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-4B-Base-GRPO"

# --- Routing ---
JS_THRESHOLD = float(os.environ.get("JS_THRESHOLD", "0.06"))
ROUTING_CASE = os.environ.get("ROUTING_CASE", "hh").lower()
ROUTER_TOP_K = 16
JS_TEMPERATURE = 1.0
ENTROPY_TOP_K = 16
ENTROPY_THRESHOLD = float(os.environ.get("ENTROPY_THRESHOLD", "0.5"))
ENTROPY_TEMPERATURE = 1.0

# When True, teacher-routed positions use argmax (greedy) instead of sampling at
# TEMPERATURE; student-routed positions are unaffected. False (default) keeps
# the original unified-sampling path so existing experiments reproduce, since
# splitting the sampling call changes the generator's random-number draw.
TEACHER_GREEDY = os.environ.get("TEACHER_GREEDY", "0") == "1"

# When TEACHER_GREEDY=0, restrict teacher sampling to the teacher's own top-k
# logits (renormalised by softmax within that top-k) before multinomial. Only
# affects teacher-routed positions; student positions still sample from the
# full vocabulary at TEMPERATURE. 0 (default) = no truncation (full-vocab
# temperature sampling, the original behavior). Ignored when TEACHER_GREEDY=1.
TEACHER_SAMPLE_TOP_K = int(os.environ.get("TEACHER_SAMPLE_TOP_K", "0"))

# Run seed for the whole experiment. None reproduces the original per-batch
# seed derivation (and the original output directory); an integer is used as
# the initial value of that derivation so different values yield different
# rollouts and the same value reproduces. When not None, the value is
# appended to the run-tag so runs with different seeds land in separate
# output directories (APPEND mode would otherwise skip them as already done).
def _parse_run_seed(val: str | None) -> int | None:
    if val is None:
        return None
    val = val.strip().lower()
    if val == "" or val == "none":
        return None
    return int(val)


RUN_SEED = _parse_run_seed(os.environ.get("SEED"))

CASE_DESCRIPTIONS = {
    "hh": "teacher_entropy high and student_entropy high",
    "hl": "teacher_entropy high and student_entropy low",
    "lh": "teacher_entropy low and student_entropy high",
    "ll": "teacher_entropy low and student_entropy low",
    "sh": "student_entropy high (teacher_entropy ignored)",
}

# --- Sampling ---
TEMPERATURE = 0.7
MAX_TOKENS = 7168

# --- Batching / parallelism ---
BATCH_SIZE = 16

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
OUT_ROOT = "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/js_entropy_cases"
PROMPT_TEMPLATE = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."

REPLACE = False
APPEND = True


def validate_config():
    if ROUTING_CASE not in CASE_DESCRIPTIONS:
        raise ValueError(
            f"Unknown ROUTING_CASE={ROUTING_CASE!r}; choose one of "
            f"{sorted(CASE_DESCRIPTIONS)}"
        )
    for name, value in (
        ("JS_TEMPERATURE", JS_TEMPERATURE),
        ("ENTROPY_TEMPERATURE", ENTROPY_TEMPERATURE),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if ROUTER_TOP_K <= 0 or ENTROPY_TOP_K <= 0:
        raise ValueError("ROUTER_TOP_K and ENTROPY_TOP_K must be positive")


def make_run_tag() -> str:
    student = Path(STUDENT_MODEL).name
    teacher = Path(TEACHER_MODEL).name
    tag = (
        f"cross_sample_{student}_TCH_{teacher}_jsth{JS_THRESHOLD}_"
        f"entcase-{ROUTING_CASE}_enth{ENTROPY_THRESHOLD}_topk{ENTROPY_TOP_K}"
    )
    if TEACHER_GREEDY:
        tag += "_tchgreedy"
    elif TEACHER_SAMPLE_TOP_K > 0:
        tag += f"_tchtopk{TEACHER_SAMPLE_TOP_K}"
    if RUN_SEED is not None:
        tag += f"_seed{RUN_SEED}"
    return tag


def make_rollout_filename(task_name: str, n: int) -> str:
    return f"{task_name.lower()}_t{TEMPERATURE}_n{n}-MNT{MAX_TOKENS}.jsonl"


# ============================================================================ #
#                              Routing helpers                                 #
# ============================================================================ #

def topk_entropy_batched(
    logits: torch.Tensor,
    top_k: int = ENTROPY_TOP_K,
    temperature: float = ENTROPY_TEMPERATURE,
) -> torch.Tensor:
    """Entropy in nats on each row's own renormalised top-k distribution."""
    k = min(top_k, logits.shape[-1])
    topk_logits = torch.topk(logits, k=k, dim=-1).values.float() / temperature
    log_probs = torch.log_softmax(topk_logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def entropy_case_mask(
    teacher_entropy: torch.Tensor,
    student_entropy: torch.Tensor,
) -> torch.Tensor:
    teacher_high = teacher_entropy > ENTROPY_THRESHOLD
    teacher_low = teacher_entropy < ENTROPY_THRESHOLD
    student_high = student_entropy > ENTROPY_THRESHOLD
    student_low = student_entropy < ENTROPY_THRESHOLD

    if ROUTING_CASE == "sh":
        return student_high
    if ROUTING_CASE == "hh":
        return teacher_high & student_high
    if ROUTING_CASE == "hl":
        return teacher_high & student_low
    if ROUTING_CASE == "lh":
        return teacher_low & student_high
    return teacher_low & student_low


def decide_sampler_batched(
    logits_stu: torch.Tensor,
    logits_tch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    js = base.js_divergence_on_union_batched(
        logits_stu,
        logits_tch,
        top_k=ROUTER_TOP_K,
        js_temperature=JS_TEMPERATURE,
    )
    student_entropy = topk_entropy_batched(logits_stu)
    teacher_entropy = topk_entropy_batched(logits_tch)
    use_teacher = (js > JS_THRESHOLD) & entropy_case_mask(
        teacher_entropy, student_entropy
    )
    return use_teacher, js, student_entropy, teacher_entropy


# ============================================================================ #
#                       Per-sequence cross-sampling                            #
# ============================================================================ #

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
    B = len(prompt_ids_list)
    if B < 1:
        raise ValueError("prompt_ids_list must not be empty")

    device = next(student.parameters()).device
    stop_token_ids = stop_token_ids if stop_token_ids is not None else set()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

    stop_ids_tensor = (
        torch.tensor(sorted(stop_token_ids), dtype=torch.long, device=device)
        if stop_token_ids else None
    )

    max_prompt_len = max(len(prompt_ids) for prompt_ids in prompt_ids_list)
    input_ids = torch.full(
        (B, max_prompt_len), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (B, max_prompt_len), dtype=torch.long, device=device
    )
    for row, prompt_ids in enumerate(prompt_ids_list):
        start = max_prompt_len - len(prompt_ids)
        input_ids[row, start:] = torch.tensor(
            prompt_ids, dtype=torch.long, device=device
        )
        attention_mask[row, start:] = 1

    stream_s = torch.cuda.Stream(device=device)
    stream_t = torch.cuda.Stream(device=device)

    def two_forwards(ids_in, attention_in, past_s, past_t):
        default_stream = torch.cuda.current_stream(device=device)
        stream_s.wait_stream(default_stream)
        stream_t.wait_stream(default_stream)
        with torch.cuda.stream(stream_s):
            out_s = student(
                input_ids=ids_in,
                attention_mask=attention_in,
                past_key_values=past_s,
                use_cache=True,
            )
        with torch.cuda.stream(stream_t):
            out_t = teacher(
                input_ids=ids_in,
                attention_mask=attention_in,
                past_key_values=past_t,
                use_cache=True,
            )
        default_stream.wait_stream(stream_s)
        default_stream.wait_stream(stream_t)
        return out_s, out_t

    done_check_every = 32
    with torch.inference_mode():
        out_s, out_t = two_forwards(input_ids, attention_mask, None, None)
        past_s = out_s.past_key_values
        past_t = out_t.past_key_values
        last_logits_s = out_s.logits[:, -1, :]
        last_logits_t = out_t.logits[:, -1, :]
        del out_s, out_t

        response_ids_buf = torch.full(
            (B, max_new_tokens), pad_id, dtype=torch.long, device=device
        )
        decisions_buf = torch.zeros(
            (B, max_new_tokens), dtype=torch.bool, device=device
        )
        js_buf = torch.zeros(
            (B, max_new_tokens), dtype=torch.float32, device=device
        )
        student_entropy_buf = torch.zeros(
            (B, max_new_tokens), dtype=torch.float32, device=device
        )
        teacher_entropy_buf = torch.zeros(
            (B, max_new_tokens), dtype=torch.float32, device=device
        )
        lengths = torch.full(
            (B,), max_new_tokens, dtype=torch.long, device=device
        )
        done = torch.zeros(B, dtype=torch.bool, device=device)
        finish_stop = torch.zeros(B, dtype=torch.bool, device=device)
        current_attention = attention_mask

        for step in range(max_new_tokens):
            use_teacher, js, student_entropy, teacher_entropy = (
                decide_sampler_batched(last_logits_s, last_logits_t)
            )

            if TEACHER_GREEDY and temperature > 0.0:
                probs_s = torch.softmax(
                    last_logits_s.float() / temperature, dim=-1
                )
                ids_s = torch.multinomial(
                    probs_s, num_samples=1, generator=generator
                ).squeeze(-1)
                ids_t = last_logits_t.argmax(dim=-1)
                next_ids = torch.where(use_teacher, ids_t, ids_s)
            elif (not TEACHER_GREEDY) and TEACHER_SAMPLE_TOP_K > 0 and temperature > 0.0:
                # Student path: unchanged full-vocab temperature sampling.
                probs_s = torch.softmax(
                    last_logits_s.float() / temperature, dim=-1
                )
                ids_s = torch.multinomial(
                    probs_s, num_samples=1, generator=generator
                ).squeeze(-1)
                # Teacher path: top-k truncation + renormalise + sample.
                t_logits = last_logits_t.float() / temperature
                k = min(TEACHER_SAMPLE_TOP_K, t_logits.shape[-1])
                topk_vals, topk_idx = torch.topk(t_logits, k=k, dim=-1)
                probs_t_topk = torch.softmax(topk_vals, dim=-1)
                sampled = torch.multinomial(
                    probs_t_topk, num_samples=1, generator=generator
                ).squeeze(-1)
                ids_t = torch.gather(topk_idx, 1, sampled.unsqueeze(-1)).squeeze(-1)
                next_ids = torch.where(use_teacher, ids_t, ids_s)
            else:
                picked_logits = torch.where(
                    use_teacher.unsqueeze(-1), last_logits_t, last_logits_s
                )
                if temperature <= 0.0:
                    next_ids = torch.argmax(picked_logits, dim=-1)
                else:
                    probabilities = torch.softmax(
                        picked_logits.float() / temperature, dim=-1
                    )
                    next_ids = torch.multinomial(
                        probabilities, num_samples=1, generator=generator
                    ).squeeze(-1)

            next_ids = torch.where(
                done, torch.full_like(next_ids, pad_id), next_ids
            )
            response_ids_buf[:, step] = next_ids
            decisions_buf[:, step] = use_teacher
            js_buf[:, step] = js
            student_entropy_buf[:, step] = student_entropy
            teacher_entropy_buf[:, step] = teacher_entropy

            if stop_ids_tensor is not None:
                stop_hit = (next_ids.unsqueeze(-1) == stop_ids_tensor).any(-1)
                new_done = stop_hit & ~done
                lengths = torch.where(
                    new_done, torch.full_like(lengths, step + 1), lengths
                )
                finish_stop |= new_done
                done |= stop_hit

            if (step + 1) % done_check_every == 0 and bool(done.all().item()):
                break
            if step + 1 == max_new_tokens:
                break

            next_input = next_ids.unsqueeze(-1)
            new_attention_column = (~done).long().unsqueeze(-1)
            current_attention = torch.cat(
                [current_attention, new_attention_column], dim=-1
            )
            out_s, out_t = two_forwards(
                next_input, current_attention, past_s, past_t
            )
            past_s = out_s.past_key_values
            past_t = out_t.past_key_values
            last_logits_s = out_s.logits[:, -1, :]
            last_logits_t = out_t.logits[:, -1, :]
            del out_s, out_t

    del past_s, past_t, last_logits_s, last_logits_t

    lengths_cpu = lengths.tolist()
    finish_stop_cpu = finish_stop.tolist()
    response_ids_cpu = response_ids_buf.tolist()
    decisions_cpu = decisions_buf.tolist()
    js_cpu = js_buf.tolist()
    student_entropy_cpu = student_entropy_buf.tolist()
    teacher_entropy_cpu = teacher_entropy_buf.tolist()

    results = []
    for row in range(B):
        length = lengths_cpu[row]
        results.append({
            "response_token_ids": response_ids_cpu[row][:length],
            "router_decisions": [
                "teacher" if decision else "student"
                for decision in decisions_cpu[row][:length]
            ],
            "js_values": js_cpu[row][:length],
            "student_topk_entropy": student_entropy_cpu[row][:length],
            "teacher_topk_entropy": teacher_entropy_cpu[row][:length],
            "finish_reason": "stop" if finish_stop_cpu[row] else "length",
        })
    return results


# ============================================================================ #
#                              Worker process                                  #
# ============================================================================ #

def worker_process(args_tuple):
    rank, gpu_id, work_units, samples, tmp_path, enable_thinking = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[TORCH_DTYPE]
    device = "cuda:0"
    student = None
    teacher = None

    try:
        print(
            f"[GPU {gpu_id}] rank={rank} | {len(work_units)} units | loading models...",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            STUDENT_MODEL, local_files_only=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        student = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL,
            dtype=torch_dtype,
            device_map=device,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        ).eval()
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL,
            dtype=torch_dtype,
            device_map=device,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        ).eval()

        stop_token_ids = base._resolve_stop_token_ids(tokenizer)
        print(
            f"[GPU {gpu_id}] models ready | stop_token_ids={sorted(stop_token_ids)}",
            flush=True,
        )

        def generator_for_batch(
            batch_units: list[tuple[int, int]],
        ) -> torch.Generator:
            seed = 0 if RUN_SEED is None else RUN_SEED
            for sample_index, sample_seed in batch_units:
                seed = (
                    seed * 1_000_003
                    + (sample_index + 1) * 1_000_003
                    + sample_seed
                ) & 0x7FFFFFFF
            generator = torch.Generator(device=device)
            generator.manual_seed(seed if seed != 0 else 1)
            return generator

        formatted_cache: dict[int, list[int]] = {}

        def prompt_ids_for(sample_index: int) -> list[int]:
            if sample_index in formatted_cache:
                return formatted_cache[sample_index]
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": samples[sample_index]["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
            formatted_cache[sample_index] = prompt_ids
            return prompt_ids

        show_progress = rank == 0
        progress = tqdm(
            total=len(work_units),
            desc=f"GPU {gpu_id} (of {len(GPUS)}, representative)",
            unit="seq",
            dynamic_ncols=True,
            mininterval=1.0,
            smoothing=0.1,
            disable=not show_progress,
        )
        start_time = time.time()
        completed_tokens = 0
        total_batches = (len(work_units) + BATCH_SIZE - 1) // BATCH_SIZE

        with open(tmp_path, "w", encoding="utf-8") as output:
            for batch_start in range(0, len(work_units), BATCH_SIZE):
                batch_index = batch_start // BATCH_SIZE + 1
                batch_units = work_units[batch_start:batch_start + BATCH_SIZE]
                prompt_ids_list = [
                    prompt_ids_for(sample_index)
                    for sample_index, _ in batch_units
                ]
                results = cross_sample_batch(
                    student,
                    teacher,
                    tokenizer,
                    prompt_ids_list,
                    max_new_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stop_token_ids=stop_token_ids,
                    generator=generator_for_batch(batch_units),
                )

                for (sample_index, seed), result in zip(batch_units, results):
                    sample = samples[sample_index]
                    response = tokenizer.decode(
                        result["response_token_ids"], skip_special_tokens=True
                    )
                    response_length = len(result["router_decisions"])
                    teacher_count = sum(
                        decision == "teacher"
                        for decision in result["router_decisions"]
                    )
                    teacher_ratio = (
                        teacher_count / response_length if response_length else 0.0
                    )
                    mean_js = (
                        sum(result["js_values"]) / response_length
                        if response_length else 0.0
                    )
                    mean_student_entropy = (
                        sum(result["student_topk_entropy"]) / response_length
                        if response_length else 0.0
                    )
                    mean_teacher_entropy = (
                        sum(result["teacher_topk_entropy"]) / response_length
                        if response_length else 0.0
                    )

                    output.write(json.dumps({
                        "example_id": sample["example_id"],
                        "prompt": sample["prompt"],
                        "answer": sample["answer"],
                        "seed": seed,
                        "response": response,
                        **result,
                        "teacher_ratio": teacher_ratio,
                        "mean_js": mean_js,
                        "mean_student_topk_entropy": mean_student_entropy,
                        "mean_teacher_topk_entropy": mean_teacher_entropy,
                        "response_length": response_length,
                        "js_threshold": JS_THRESHOLD,
                        "entropy_case": ROUTING_CASE,
                        "entropy_threshold": ENTROPY_THRESHOLD,
                        "entropy_top_k": ENTROPY_TOP_K,
                        "entropy_temperature": ENTROPY_TEMPERATURE,
                        "teacher_greedy": TEACHER_GREEDY,
                        "teacher_sample_top_k": TEACHER_SAMPLE_TOP_K,
                        "run_seed": RUN_SEED,
                    }, ensure_ascii=False) + "\n")
                    completed_tokens += response_length
                    progress.update(1)

                if show_progress:
                    elapsed = time.time() - start_time
                    token_rate = completed_tokens / elapsed if elapsed > 0 else 0.0
                    progress.set_postfix(
                        bch=f"{batch_index}/{total_batches}",
                        tok_per_s=f"{token_rate:.0f}",
                        refresh=True,
                    )
                output.flush()

        progress.close()
        print(
            f"[GPU {gpu_id}] done {len(work_units)} units, "
            f"{completed_tokens} tokens, {(time.time() - start_time) / 60:.1f} min.",
            flush=True,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        del student, teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================ #
#                              Generation driver                               #
# ============================================================================ #

def run_generation(enable_thinking: bool):
    print("=" * 76)
    print("  Cross-sampling generation: JS gate + entropy case")
    print(f"  student  = {STUDENT_MODEL}")
    print(f"  teacher  = {TEACHER_MODEL}")
    print(
        f"  routing  = JS>{JS_THRESHOLD} and ({CASE_DESCRIPTIONS[ROUTING_CASE]})"
    )
    print(
        f"  JS       = union top-{ROUTER_TOP_K}, T={JS_TEMPERATURE}; "
        f"entropy = own top-{ENTROPY_TOP_K}, T={ENTROPY_TEMPERATURE}"
    )
    if TEACHER_GREEDY:
        print(f"  sampling = teacher: argmax (greedy), student: T={TEMPERATURE}")
    elif TEACHER_SAMPLE_TOP_K > 0:
        print(f"  sampling = teacher: top-{TEACHER_SAMPLE_TOP_K} + T={TEMPERATURE}, "
              f"student: T={TEMPERATURE} (full vocab)")
    else:
        print(f"  sampling = T={TEMPERATURE} (both routes, full vocab)")
    print(f"  max_tokens={MAX_TOKENS}")
    print(f"  batch    = {BATCH_SIZE}/GPU")
    print(f"  GPUs     = {GPUS}")
    seed_display = "none (original derivation, reproduces prior runs)" if RUN_SEED is None else RUN_SEED
    print(f"  seed     = {seed_display}")
    print("=" * 76)

    output_dir = Path(OUT_ROOT) / make_run_tag()
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        task_name = task["name"]
        n = task["N"]
        output_path = output_dir / make_rollout_filename(task_name, n)
        print(f"\n--- Task: {task_name} (N={n}) -> {output_path}")

        if output_path.exists() and not REPLACE and not APPEND:
            print("  exists and REPLACE=APPEND=False -> skip.")
            continue

        existing = []
        done_pairs: set[tuple[int, int]] = set()
        if APPEND and not REPLACE and output_path.exists():
            print(f"  APPEND: reading existing rollouts from {output_path}")
            with output_path.open(encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    existing.append(item)
                    done_pairs.add((int(item["example_id"]), int(item["seed"])))
            print(f"  APPEND: {len(done_pairs)} (sample,seed) pairs already done.")

        samples = base.load_samples(task["path"])
        for sample in samples:
            sample["prompt"] = PROMPT_TEMPLATE.format(problem=sample["prompt"])

        work_units = [
            (sample_index, seed)
            for sample_index in range(len(samples))
            for seed in range(n)
            if (sample_index, seed) not in done_pairs
        ]
        if not work_units:
            print(f"  all {len(samples) * n} units already done; nothing to generate.")
            continue

        print(f"  generating {len(work_units)} units across {len(GPUS)} GPU(s).")
        chunks = base.split_units(work_units, len(GPUS))
        tmp_paths = [
            output_dir / f"_tmp_gpu{rank}.jsonl" for rank in range(len(GPUS))
        ]
        worker_args = [
            (
                rank,
                gpu_id,
                chunks[rank],
                samples,
                str(tmp_paths[rank]),
                enable_thinking,
            )
            for rank, gpu_id in enumerate(GPUS)
            if chunks[rank]
        ]

        context = mp.get_context("spawn")
        processes = []
        for args in worker_args:
            process = context.Process(target=worker_process, args=(args,))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(
                f"{len(failed)} worker(s) failed with exit codes: "
                + ", ".join(str(process.exitcode) for process in failed)
            )

        all_results = list(existing)
        for tmp_path in tmp_paths:
            if not tmp_path.exists():
                continue
            with tmp_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_results.append(json.loads(line))
            tmp_path.unlink()

        expected_count = len(samples) * n
        if len(all_results) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} rollouts but collected {len(all_results)}"
            )
        all_results.sort(key=lambda item: (int(item["example_id"]), int(item["seed"])))
        with output_path.open("w", encoding="utf-8") as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  wrote {len(all_results)} rollouts to {output_path}")


# ============================================================================ #
#                                  Grading                                     #
# ============================================================================ #

def grade_file(file_path: Path) -> dict:
    base.JS_THRESHOLD = JS_THRESHOLD
    summary = base.grade_file(file_path)
    if summary:
        summary.update({
            "routing": "js_and_entropy_case",
            "entropy_case": ROUTING_CASE,
            "entropy_case_description": CASE_DESCRIPTIONS[ROUTING_CASE],
            "entropy_threshold": ENTROPY_THRESHOLD,
            "entropy_top_k": ENTROPY_TOP_K,
            "entropy_temperature": ENTROPY_TEMPERATURE,
        })
    return summary


def run_grading(grade_file_override: str | None = None):
    if grade_file_override:
        files = [Path(grade_file_override)]
        combined_dir = files[0].parent
    else:
        combined_dir = Path(OUT_ROOT) / make_run_tag()
        if not combined_dir.exists():
            print(f"Output dir does not exist: {combined_dir}")
            return
        files = sorted(combined_dir.glob("*.jsonl"))

    summaries = []
    for file_path in files:
        if not file_path.exists():
            print(f"Skip missing: {file_path}")
            continue
        summary = grade_file(file_path)
        if not summary:
            continue
        summaries.append(summary)
        summary_path = file_path.with_name(file_path.stem + "_grading.json")
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  summary saved to {summary_path}")

    if summaries:
        combined = [
            {key: value for key, value in summary.items() if key != "per_problem"}
            for summary in summaries
        ]
        combined_path = combined_dir / "grading_summary.json"
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"Combined summary saved to {combined_path}")


def main():
    validate_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation.")
    parser.add_argument("--skip-grade", action="store_true", help="Skip grading.")
    parser.add_argument("--grade-file", default=None)
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true"
    )
    thinking_group.add_argument(
        "--disable-thinking", dest="enable_thinking", action="store_false"
    )
    parser.set_defaults(enable_thinking=ENABLE_THINKING)
    args = parser.parse_args()

    if args.grade_file:
        args.skip_gen = True
    if not args.skip_gen:
        print("\n" + "=" * 76)
        print("  STEP 1 / 2: Cross-sampling generation")
        print("=" * 76)
        run_generation(enable_thinking=args.enable_thinking)
    if not args.skip_grade:
        print("\n" + "=" * 76)
        print("  STEP 2 / 2: Grading")
        print("=" * 76)
        run_grading(grade_file_override=args.grade_file)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
