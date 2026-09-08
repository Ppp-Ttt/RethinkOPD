"""
Compute the per-position probability difference (teacher - student) of the
actually-generated token under each model's own top-k renormalised
distribution, for an existing cross-sample JSONL file.

The saved response_token_ids are replayed without generating new tokens. For
each response position i, the actually-generated token t = response_token_ids[i]
is looked up in each model's top-k token set (top-k of that model's logits,
renormalised by softmax after dividing by ENTROPY_TEMPERATURE):

  - student_topk_prob[i] = student's renormalised prob of t (0 if t not in
    student's top-k)
  - teacher_topk_prob[i] = teacher's renormalised prob of t (0 if t not in
    teacher's top-k)
  - topk_prob_diff[i]     = teacher_topk_prob[i] - student_topk_prob[i]

Student and teacher share the same vocabulary, so token ids are compared
directly.

The output preserves every input field and adds:
  - student_topk_prob / teacher_topk_prob / topk_prob_diff
  - mean_student_topk_prob / mean_teacher_topk_prob / mean_topk_prob_diff
  - prob_diff_top_k / prob_diff_temperature

Edit the global parameters below, then run:
  python compute_topk_prob_diff.py

The input file is never modified.
"""

# ============================================================================ #
#                  Global parameters - edit here before running                #
# ============================================================================ #

# STUDENT_MODEL = "/mmu_cd_ssd/pengtiantian/projects/OPD/checkpoint_rerun0711/1.7B-4B_JS-ADD-FKL_token_reward_direct_DAPO-Math-17k_Qwen3-1.7B-Base_Qwen3-4B-Base-GRPO_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_union-rw_js_add_fkl-2026-07-12_12-24-47/global_step_140/1.7B-4B_JS-ADD-FKL_step140"
STUDENT_MODEL = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-1.7B-Base"
TEACHER_MODEL = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-4B-Base-GRPO"

INPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.01_topk16/amc23_t0.7_n8-MNT7168.jsonl"
)
OUTPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.01_topk16/Base_top16_prob_diff.jsonl"
)

# Top-k used to renormalise each model's distribution before reading off the
# probability of the actually-generated token.
TOP_K = 16
ENTROPY_TEMPERATURE = 1.0

# These must match the settings used to generate the rollout file.
APPLY_CHAT_TEMPLATE = True
ENABLE_THINKING = False

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
TORCH_DTYPE = "bfloat16"

# Bounds the largest logits tensor per model. In bfloat16, 32768 tokens with a
# 151936-token vocabulary require about 9.3 GiB for logits.
BATCH_TOKEN_BUDGET = 32768

# Refuse to replace an existing output unless explicitly enabled here.
REPLACE = False


# ============================================================================ #
#                                  Implementation                              #
# ============================================================================ #

import gc
import json
import multiprocessing as mp
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for field in ("prompt", "response_token_ids"):
                if field not in record:
                    raise ValueError(f"{path}:{line_no}: missing required field {field!r}")
            if not isinstance(record["response_token_ids"], list):
                raise ValueError(f"{path}:{line_no}: response_token_ids must be a list")
            records.append(record)
    return records


def build_sequences(records: list[dict], tokenizer, apply_chat_template: bool,
                    enable_thinking: bool) -> list[dict]:
    sequences = []
    for record_index, record in enumerate(records):
        if apply_chat_template:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": record["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        else:
            prompt_ids = tokenizer.encode(record["prompt"], add_special_tokens=True)

        if not prompt_ids:
            raise ValueError(f"Record {record_index} produced an empty prompt token sequence.")

        response_ids = [int(token_id) for token_id in record["response_token_ids"]]
        sequences.append({
            "record_index": record_index,
            "full_ids": prompt_ids + response_ids,
            "prompt_len": len(prompt_ids),
            "response_token_ids": response_ids,
        })
    return sequences


def make_batches(sequences: list[dict], token_budget: int) -> list[list[dict]]:
    """Pack similarly sized sequences under max_seq_len * batch_size budget."""
    sorted_sequences = sorted(sequences, key=lambda seq: len(seq["full_ids"]))
    batches: list[list[dict]] = []
    batch: list[dict] = []
    batch_max_len = 0

    for sequence in sorted_sequences:
        sequence_len = len(sequence["full_ids"])
        new_max_len = max(batch_max_len, sequence_len)
        if batch and new_max_len * (len(batch) + 1) > token_budget:
            batches.append(batch)
            batch = [sequence]
            batch_max_len = sequence_len
        else:
            batch.append(sequence)
            batch_max_len = new_max_len

    if batch:
        batches.append(batch)
    return batches


def topk_prob_diff_from_logits(student_logits: torch.Tensor,
                               teacher_logits: torch.Tensor,
                               actual_token_ids: torch.Tensor, top_k: int,
                               temperature: float):
    """Return (student_prob, teacher_prob, diff) lists for each row (position).

    actual_token_ids: [L] long tensor of the actually-generated token ids at
    each response position. For each position, the prob of that token under
    each model's own top-k renormalised distribution is read off; 0 if the
    token is outside the model's top-k.
    """
    if student_logits.shape[0] == 0:
        return [], [], []

    k = min(top_k, student_logits.shape[-1], teacher_logits.shape[-1])

    s_vals, s_ids = torch.topk(student_logits, k=k, dim=-1)
    t_vals, t_ids = torch.topk(teacher_logits, k=k, dim=-1)
    s_probs = torch.softmax(s_vals.float() / temperature, dim=-1)  # [L, k]
    t_probs = torch.softmax(t_vals.float() / temperature, dim=-1)  # [L, k]

    actual = actual_token_ids.unsqueeze(1)  # [L, 1]
    s_match = (s_ids == actual).to(s_probs.dtype)  # [L, k]
    t_match = (t_ids == actual).to(t_probs.dtype)  # [L, k]
    # At most one column matches per row, so summing is safe.
    student_prob = (s_probs * s_match).sum(dim=1)  # [L]
    teacher_prob = (t_probs * t_match).sum(dim=1)  # [L]
    diff = teacher_prob - student_prob

    return (
        student_prob.cpu().tolist(),
        teacher_prob.cpu().tolist(),
        diff.cpu().tolist(),
    )


def score_batch(student, teacher, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, batch: list[dict], max_len: int,
                top_k: int, entropy_temperature: float, device: str):
    with torch.inference_mode():
        s_out = student(input_ids=input_ids, attention_mask=attention_mask)
        s_logits = s_out.logits
        t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
        t_logits = t_out.logits

    results = []
    for row, sequence in enumerate(batch):
        response_len = len(sequence["response_token_ids"])
        if response_len == 0:
            results.append(([], [], []))
            continue

        left_pad = max_len - len(sequence["full_ids"])
        # Logits at prompt_len - 1 predict response token 0, keeping position i
        # aligned with response_token_ids[i].
        start = left_pad + sequence["prompt_len"] - 1
        end = start + response_len
        s_slice = s_logits[row, start:end, :]
        t_slice = t_logits[row, start:end, :]
        actual_ids = torch.tensor(
            sequence["response_token_ids"], dtype=torch.long, device=device
        )
        results.append(topk_prob_diff_from_logits(
            s_slice, t_slice, actual_ids, top_k, entropy_temperature
        ))

    del s_out, t_out, s_logits, t_logits
    return results


def load_model(model_path: str, device: str, torch_dtype: torch.dtype):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=device,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).eval()


def score_on_device(rank: int, gpu_id: int, sequences: list[dict],
                    pad_token_id: int, tmp_path: str, student_model_path: str,
                    teacher_model_path: str, top_k: int,
                    entropy_temperature: float, torch_dtype_name: str,
                    batch_token_budget: int):
    device = f"cuda:{gpu_id}"
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[torch_dtype_name]

    student = None
    teacher = None
    try:
        print(
            f"[GPU {gpu_id}] rank={rank} | {len(sequences)} sequences | loading models...",
            flush=True,
        )
        student = load_model(student_model_path, device, torch_dtype)
        teacher = load_model(teacher_model_path, device, torch_dtype)

        student_vocab = student.get_output_embeddings().weight.shape[0]
        teacher_vocab = teacher.get_output_embeddings().weight.shape[0]
        if student_vocab != teacher_vocab:
            raise ValueError(
                f"Student/teacher vocabulary sizes differ: {student_vocab} != {teacher_vocab}"
            )

        batches = make_batches(sequences, batch_token_budget)
        with open(tmp_path, "x", encoding="utf-8") as out_f:
            for batch_index, batch in enumerate(batches, start=1):
                max_len = max(len(sequence["full_ids"]) for sequence in batch)
                print(
                    f"[GPU {gpu_id}] batch {batch_index}/{len(batches)} "
                    f"(size={len(batch)}, max_len={max_len})",
                    flush=True,
                )

                input_ids = torch.full(
                    (len(batch), max_len),
                    pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros(
                    (len(batch), max_len), dtype=torch.long, device=device
                )
                for row, sequence in enumerate(batch):
                    ids = sequence["full_ids"]
                    start = max_len - len(ids)
                    input_ids[row, start:] = torch.tensor(ids, dtype=torch.long, device=device)
                    attention_mask[row, start:] = 1

                batch_results = score_batch(
                    student, teacher, input_ids, attention_mask, batch,
                    max_len, top_k, entropy_temperature, device,
                )
                torch.cuda.empty_cache()

                for sequence, (s_prob, t_prob, diff) in zip(batch, batch_results):
                    response_len = len(sequence["response_token_ids"])
                    if len(s_prob) != response_len or len(t_prob) != response_len or len(diff) != response_len:
                        raise RuntimeError(
                            f"Metric length mismatch for record {sequence['record_index']}"
                        )
                    out_f.write(json.dumps({
                        "record_index": sequence["record_index"],
                        "student_topk_prob": s_prob,
                        "teacher_topk_prob": t_prob,
                        "topk_prob_diff": diff,
                    }, ensure_ascii=False) + "\n")
                out_f.flush()

                del input_ids, attention_mask, batch_results
                torch.cuda.empty_cache()

        print(f"[GPU {gpu_id}] done.", flush=True)
    finally:
        del student, teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    input_path = Path(INPUT_PATH).resolve()
    output_path = Path(OUTPUT_PATH).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input rollout file does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Output path must differ from input path.")
    if output_path.exists() and not REPLACE:
        raise FileExistsError(f"Output exists; set REPLACE=True to replace it: {output_path}")
    if BATCH_TOKEN_BUDGET <= 0:
        raise ValueError("BATCH_TOKEN_BUDGET must be positive.")
    if TOP_K <= 0 or ENTROPY_TEMPERATURE <= 0:
        raise ValueError("TOP_K and ENTROPY_TEMPERATURE must be positive.")
    if not GPUS or len(GPUS) != len(set(GPUS)) or min(GPUS) < 0:
        raise ValueError("GPUS must contain unique non-negative GPU IDs.")

    available_gpus = torch.cuda.device_count()
    invalid_gpus = [gpu_id for gpu_id in GPUS if gpu_id >= available_gpus]
    if invalid_gpus:
        raise RuntimeError(
            f"Unavailable GPU IDs {invalid_gpus}; torch sees {available_gpus} GPU(s)."
        )

    print("=" * 72)
    print("  Cross-sample student/teacher top-k prob diff (teacher - student)")
    print(f"  input       = {input_path}")
    print(f"  output      = {output_path}")
    print(f"  student     = {STUDENT_MODEL}")
    print(f"  teacher     = {TEACHER_MODEL}")
    print(f"  top-k       = {TOP_K}, T={ENTROPY_TEMPERATURE}")
    print(f"  chat        = apply_template={APPLY_CHAT_TEMPLATE}, thinking={ENABLE_THINKING}")
    print(f"  GPUs        = {GPUS}")
    print(f"  token budget= {BATCH_TOKEN_BUDGET}/GPU")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_MODEL, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

    records = load_records(input_path)
    print(f"Loaded {len(records)} rollout records.")
    sequences = build_sequences(
        records, tokenizer, APPLY_CHAT_TEMPLATE, ENABLE_THINKING
    )
    if not sequences:
        raise ValueError("Input file contains no rollout records.")

    n_workers = min(len(GPUS), len(sequences))
    gpu_ids = GPUS[:n_workers]
    chunks = [sequences[index::n_workers] for index in range(n_workers)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_paths = [
        output_path.with_name(f".{output_path.name}.gpu{rank}.tmp")
        for rank in range(n_workers)
    ]
    stale_tmp = [path for path in tmp_paths if path.exists()]
    if stale_tmp:
        raise FileExistsError(
            "Temporary output exists from another or interrupted run: "
            + ", ".join(str(path) for path in stale_tmp)
        )

    ctx = mp.get_context("spawn")
    processes = []
    for rank, (gpu_id, chunk, tmp_path) in enumerate(zip(gpu_ids, chunks, tmp_paths)):
        process = ctx.Process(
            target=score_on_device,
            args=(
                rank, gpu_id, chunk, tokenizer.pad_token_id, str(tmp_path),
                STUDENT_MODEL, TEACHER_MODEL, TOP_K,
                ENTROPY_TEMPERATURE, TORCH_DTYPE, BATCH_TOKEN_BUDGET,
            ),
        )
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

    metrics_by_index = {}
    for tmp_path in tmp_paths:
        with tmp_path.open(encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                record_index = int(result.pop("record_index"))
                if record_index in metrics_by_index:
                    raise RuntimeError(f"Duplicate metric result for record {record_index}")
                metrics_by_index[record_index] = result

    missing = sorted(set(range(len(records))) - set(metrics_by_index))
    if missing:
        raise RuntimeError(f"Missing metric results for {len(missing)} record(s): {missing[:10]}")

    with output_path.open("w", encoding="utf-8") as out_f:
        for record_index, record in enumerate(records):
            result = dict(record)
            metrics = metrics_by_index[record_index]
            result.update(metrics)
            s_prob = metrics["student_topk_prob"]
            t_prob = metrics["teacher_topk_prob"]
            diff = metrics["topk_prob_diff"]
            result.update({
                "mean_student_topk_prob": (sum(s_prob) / len(s_prob)) if s_prob else None,
                "mean_teacher_topk_prob": (sum(t_prob) / len(t_prob)) if t_prob else None,
                "mean_topk_prob_diff": (sum(diff) / len(diff)) if diff else None,
                "prob_diff_top_k": TOP_K,
                "prob_diff_temperature": ENTROPY_TEMPERATURE,
            })
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    for tmp_path in tmp_paths:
        tmp_path.unlink()

    print(f"Done. Wrote {len(records)} enriched records to {output_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
