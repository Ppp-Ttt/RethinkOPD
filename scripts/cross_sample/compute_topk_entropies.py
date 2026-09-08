"""
Compute student/teacher top-k entropy for an existing cross-sample JSONL file.

The saved response_token_ids are replayed without generating new tokens. For
each response position, both models' next-token logits are restricted to that
model's own top-k tokens, renormalised, and converted to entropy in nats.

The output preserves every input field and adds:
  - student_topk_entropy / teacher_topk_entropy: per-token entropy arrays
  - mean_student_topk_entropy / mean_teacher_topk_entropy
  - entropy_top_k / entropy_temperature

Edit the global parameters below, then run:
  python compute_topk_entropies.py

The input file is never modified.
"""

# ============================================================================ #
#                  Global parameters - edit here before running                #
# ============================================================================ #

STUDENT_MODEL = "/mmu_cd_ssd/pengtiantian/projects/OPD/checkpoint_rerun0820/1.7B-4B_SPARSE-RKL20%_token_reward_direct_DAPO-Math-17k_Qwen3-1.7B-Base_Qwen3-4B-Base-GRPO_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_union-rw_sparse_rkl-2026-08-20_20-08-24/global_step_279/1.7B-4B_SPARSE-RKL20%_step279"
TEACHER_MODEL = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-4B-Base-GRPO"

INPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample_0820/cross_sample_1.7B-4B_SPARSE-RKL20%_step279_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_1.7B-4B_SPARSE-RKL20%_step279_TCH_Qwen3-4B-Base-GRPO_jsth0.1_topk16/amc23_t0.7_n8-MNT7168.jsonl"
)
OUTPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample_0820/cross_sample_1.7B-4B_SPARSE-RKL20%_step279_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_1.7B-4B_SPARSE-RKL20%_step279_TCH_Qwen3-4B-Base-GRPO_jsth0.1_topk16/amc23_t0.7_n8-MNT7168_top16_entropy.jsonl"
)

# Entropy is computed after renormalising each model's own top-16 logits.
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


def topk_entropy_from_logits(logits: torch.Tensor, top_k: int,
                             temperature: float) -> list[float]:
    """Return entropy in nats after renormalising over each row's top-k logits."""
    if logits.shape[0] == 0:
        return []
    k = min(top_k, logits.shape[-1])
    topk_logits = torch.topk(logits, k=k, dim=-1).values.float() / temperature
    log_probs = torch.log_softmax(topk_logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return entropy.cpu().tolist()


def score_model(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                batch: list[dict], max_len: int, top_k: int,
                entropy_temperature: float) -> list[list[float]]:
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits

    entropies = []
    for row, sequence in enumerate(batch):
        response_len = len(sequence["response_token_ids"])
        if response_len == 0:
            entropies.append([])
            continue

        left_pad = max_len - len(sequence["full_ids"])
        # Logits at prompt_len - 1 predict response token 0. This keeps entropy
        # position i aligned with response_token_ids[i] and js_values[i].
        start = left_pad + sequence["prompt_len"] - 1
        end = start + response_len
        response_logits = logits[row, start:end, :]
        entropies.append(topk_entropy_from_logits(
            response_logits, top_k, entropy_temperature
        ))

    del output, logits
    return entropies


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

                student_entropy = score_model(
                    student, input_ids, attention_mask, batch, max_len,
                    top_k, entropy_temperature,
                )
                torch.cuda.empty_cache()
                teacher_entropy = score_model(
                    teacher, input_ids, attention_mask, batch, max_len,
                    top_k, entropy_temperature,
                )

                for sequence, student_values, teacher_values in zip(
                    batch, student_entropy, teacher_entropy
                ):
                    response_len = len(sequence["response_token_ids"])
                    if len(student_values) != response_len or len(teacher_values) != response_len:
                        raise RuntimeError(
                            f"Entropy length mismatch for record {sequence['record_index']}"
                        )
                    out_f.write(json.dumps({
                        "record_index": sequence["record_index"],
                        "student_topk_entropy": student_values,
                        "teacher_topk_entropy": teacher_values,
                    }, ensure_ascii=False) + "\n")
                out_f.flush()

                del input_ids, attention_mask, student_entropy, teacher_entropy
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
    print("  Cross-sample student/teacher top-k entropy")
    print(f"  input       = {input_path}")
    print(f"  output      = {output_path}")
    print(f"  student     = {STUDENT_MODEL}")
    print(f"  teacher     = {TEACHER_MODEL}")
    print(f"  entropy     = own top-{TOP_K}, T={ENTROPY_TEMPERATURE}, units=nats")
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

    entropy_by_index = {}
    for tmp_path in tmp_paths:
        with tmp_path.open(encoding="utf-8") as f:
            for line in f:
                result = json.loads(line)
                record_index = int(result.pop("record_index"))
                if record_index in entropy_by_index:
                    raise RuntimeError(f"Duplicate entropy result for record {record_index}")
                entropy_by_index[record_index] = result

    missing = sorted(set(range(len(records))) - set(entropy_by_index))
    if missing:
        raise RuntimeError(f"Missing entropy results for {len(missing)} record(s): {missing[:10]}")

    with output_path.open("w", encoding="utf-8") as out_f:
        for record_index, record in enumerate(records):
            result = dict(record)
            entropy = entropy_by_index[record_index]
            result.update(entropy)
            result.update({
                "mean_student_topk_entropy": (
                    sum(entropy["student_topk_entropy"]) / len(entropy["student_topk_entropy"])
                    if entropy["student_topk_entropy"] else None
                ),
                "mean_teacher_topk_entropy": (
                    sum(entropy["teacher_topk_entropy"]) / len(entropy["teacher_topk_entropy"])
                    if entropy["teacher_topk_entropy"] else None
                ),
                "entropy_top_k": TOP_K,
                "entropy_temperature": ENTROPY_TEMPERATURE,
            })
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    for tmp_path in tmp_paths:
        tmp_path.unlink()

    print(f"Done. Wrote {len(records)} enriched records to {output_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
