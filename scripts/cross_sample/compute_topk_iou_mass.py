"""
Compute top-k IoU and intersection probability mass between student and teacher
for each response position in an existing cross-sample JSONL file.

The saved response_token_ids are replayed without generating new tokens. For
each response position, each model's next-token logits are restricted to that
model's own top-k tokens (k = TOP_K), renormalised via softmax after dividing
by ENTROPY_TEMPERATURE. Three per-position metrics are then computed from the
two top-k token-id sets S (student) and T (teacher):

  - topk_iou                 = |S ∩ T| / |S ∪ T|
  - student_intersection_mass = sum of student probs over S ∩ T
  - teacher_intersection_mass = sum of teacher probs over S ∩ T

Student and teacher share the same vocabulary, so top-k token ids are compared
directly.

The output preserves every input field and adds:
  - topk_iou / student_intersection_mass / teacher_intersection_mass
  - mean_topk_iou / mean_student_intersection_mass / mean_teacher_intersection_mass
  - iou_top_k / iou_temperature

Edit the global parameters below, then run:
  python compute_topk_iou_mass.py

The input file is never modified.
"""

# ============================================================================ #
#                  Global parameters - edit here before running                #
# ============================================================================ #

STUDENT_MODEL = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-1.7B-Base"
TEACHER_MODEL = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-4B-Base-GRPO"

INPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168.jsonl"
)
OUTPUT_PATH = (
    "/mmu_cd_ssd/pengtiantian/projects/OPD/results/cross_sample/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_js/cross_sample_Qwen3-1.7B-Base_TCH_Qwen3-4B-Base-GRPO_jsth0.06_topk16/amc23_t0.7_n8-MNT7168_top16_iou_mass.jsonl"
)

# Top-k used for both IoU and probability-mass computation.
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


def topk_iou_mass_from_logits(student_logits: torch.Tensor,
                              teacher_logits: torch.Tensor, top_k: int,
                              temperature: float):
    """Return (iou, student_mass, teacher_mass) lists for each row (position)."""
    if student_logits.shape[0] == 0:
        return [], [], []

    k = min(top_k, student_logits.shape[-1], teacher_logits.shape[-1])

    # Top-k ids/values per model; probs renormalised over the top-k logits.
    s_vals, s_ids = torch.topk(student_logits, k=k, dim=-1)
    t_vals, t_ids = torch.topk(teacher_logits, k=k, dim=-1)
    s_probs = torch.softmax(s_vals.float() / temperature, dim=-1)
    t_probs = torch.softmax(t_vals.float() / temperature, dim=-1)

    # match[l, i, j] = (student token i == teacher token j) at position l.
    match = (s_ids.unsqueeze(2) == t_ids.unsqueeze(1))  # [L, k, k] bool
    inter_size = match.any(dim=2).sum(dim=1).to(torch.float32)  # [L]
    union_size = 2 * k - inter_size
    iou = torch.where(
        union_size > 0, inter_size / union_size,
        torch.zeros_like(inter_size),
    )

    student_mask = match.any(dim=2).to(s_probs.dtype)  # [L, k]
    teacher_mask = match.any(dim=1).to(t_probs.dtype)  # [L, k]
    student_mass = (s_probs * student_mask).sum(dim=1)
    teacher_mass = (t_probs * teacher_mask).sum(dim=1)

    return (
        iou.cpu().tolist(),
        student_mass.cpu().tolist(),
        teacher_mass.cpu().tolist(),
    )


def score_batch(student, teacher, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, batch: list[dict], max_len: int,
                top_k: int, entropy_temperature: float):
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
        results.append(topk_iou_mass_from_logits(
            s_slice, t_slice, top_k, entropy_temperature
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
                    max_len, top_k, entropy_temperature,
                )
                torch.cuda.empty_cache()

                for sequence, (iou_vals, s_mass, t_mass) in zip(batch, batch_results):
                    response_len = len(sequence["response_token_ids"])
                    if len(iou_vals) != response_len or len(s_mass) != response_len or len(t_mass) != response_len:
                        raise RuntimeError(
                            f"Metric length mismatch for record {sequence['record_index']}"
                        )
                    out_f.write(json.dumps({
                        "record_index": sequence["record_index"],
                        "topk_iou": iou_vals,
                        "student_intersection_mass": s_mass,
                        "teacher_intersection_mass": t_mass,
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
    print("  Cross-sample student/teacher top-k IoU + intersection mass")
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
            iou_vals = metrics["topk_iou"]
            s_mass = metrics["student_intersection_mass"]
            t_mass = metrics["teacher_intersection_mass"]
            result.update({
                "mean_topk_iou": (sum(iou_vals) / len(iou_vals)) if iou_vals else None,
                "mean_student_intersection_mass": (sum(s_mass) / len(s_mass)) if s_mass else None,
                "mean_teacher_intersection_mass": (sum(t_mass) / len(t_mass)) if t_mass else None,
                "iou_top_k": TOP_K,
                "iou_temperature": ENTROPY_TEMPERATURE,
            })
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    for tmp_path in tmp_paths:
        tmp_path.unlink()

    print(f"Done. Wrote {len(records)} enriched records to {output_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
