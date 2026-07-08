import os
import sys
import json
import re
import math
import concurrent.futures
import multiprocessing
import gc
import torch
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import grade_answer_verl

try:
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    destroy_model_parallel = None
try:
    from vllm.distributed.parallel_state import destroy_distributed_environment
except ImportError:
    destroy_distributed_environment = None

# =========================================================================== #
#   Single-checkpoint variant of gen_vllm_grade_steps.py                      #
#                                                                             #
#   Use this when you already have ONE specific weight directory and want to  #
#   tag it with a step number, generate rollouts, grade them, and merge the   #
#   result into a grading_results.json under `{OUT_DIR_NAME}/{MODEL_NAME}`.   #
#                                                                             #
#   The resulting JSON is keyed by `step{STEP}` exactly like the steps        #
#   pipeline, so a single grading_results.json can hold mixed entries from    #
#   both the steps pipeline and this single-checkpoint pipeline.              #
# =========================================================================== #

# --- Target checkpoint (edit here before running) ---
# Absolute path to the weight directory that vLLM should load.
WEIGHT_DIR = "/mmu_cd_ssd/pengtiantian/projects/OPD/checkpoint_newFKL/Qwen3-4B-Base_OPD_by_Qwen3-8B-Base-GRPO_JS-ADD-FKL_token_reward_direct_DAPO-Math-17k_Qwen3-4B-Base_Qwen3-8B-Base-GRPO_step279_7168-T_1.0-Tch_1.0-n_4-mbs_64-topk_16-topk_strategy_union-rw_js_add_fkl-2026-06-28_19-04-05/global_step_279/Qwen3-4B-Base_OPD_by_Qwen3-8B-Base-GRPO_JS-ADD-FKL_step279"
# Step tag used in output filenames and as the `stepN` key in grading_results.json.
STEP = 279
# Name of the model bucket. Output goes to `{OUT_DIR_NAME}/{MODEL_NAME}/...`.
# If you want results merged into an existing grading_results.json produced by
# the steps pipeline, set this to that pipeline's `model_name`.
MODEL_NAME = "Qwen3-4B-Base_OPD_by_Qwen3-8B-Base-GRPO_JS-ADD-FKL"

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

DATA_DIR = "../data"
TASKS = [
    {"name": "AIME24",   "path": f"{DATA_DIR}/AIME24/test.parquet",   "N": 128},
    {"name": "AIME25",   "path": f"{DATA_DIR}/AIME25/test.parquet",   "N": 16},
    {"name": "AMC23",    "path": f"{DATA_DIR}/AMC23/test.parquet",    "N": 128},
    {"name": "MATH500", "path": f"{DATA_DIR}/MATH-500/test.parquet", "N": 16},
    {"name": "Minerva",   "path": f"{DATA_DIR}/Minerva/test.parquet",   "N": 16},
    {"name": "Olympia",   "path": f"{DATA_DIR}/Olympiad-Bench/test.parquet",   "N": 16},
]

MAX_TOKENS  = 7168
TEMPERATURE = 0.7
TOP_P       = 0.95
REPLACE     = False
APPEND      = True

# --- Pipeline switches ---
SKIP_GEN              = False
SKIP_GRADE            = False
ENABLE_THINKING       = False
ENABLE_MODEL_VERIFIER = False
DELETE_ROLLOUTS       = True

# --- Grading ---
LENGTH_TOKENIZER_PATH = "/mmu_cd_ssd/pengtiantian/projects/OPD/models/Qwen3-1.7B"
VERIFIER_MODEL_PATH   = "../../model/CompassVerifier-3B"

# --- Derived ---
OUT_DIR_NAME    = "/mmu_cd_ssd/pengtiantian/projects/OPD/eval_output"
PROMPT_TEMPLATE = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."
K_VALUES = [1,2,3,4,5,6,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192,224,256,320,384,448,512]

# =========================================================================== #
#                          CV prompt (grading)                                #
# =========================================================================== #
CV_PROMPT = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>:
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""

# =========================================================================== #
#                          Generation helpers                                 #
# =========================================================================== #

def load_samples(filepath: str):
    df = pd.read_parquet(filepath)
    if any(tag in filepath for tag in ("BRUMO25", "CMIMC25", "HMMT25")):
        samples = [
            {"example_id": i, "prompt": df.at[i, "problem"].strip(), "answer": df.at[i, "answer"].strip()}
            for i in range(len(df))
        ]
    else:
        samples = [
            {"example_id": i, "prompt": df.at[i, "prompt"][0]["content"].strip(), "answer": df.at[i, "reward_model"]["ground_truth"].strip()}
            for i in range(len(df))
        ]
    print(f"Total unique samples: {len(samples)}")
    return samples


def split_rollout_ids(rollout_ids: list[int], num_workers: int):
    chunks = [[] for _ in range(num_workers)]
    for idx, rollout_id in enumerate(rollout_ids):
        chunks[idx % num_workers].append(rollout_id)
    return chunks


def worker_process(args_tuple):
    model_path, samples, rollout_id_list, gpu_id, enable_thinking = args_tuple

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    results = []
    llm = None
    stop_token_ids = []

    try:
        print(
            f"[GPU {gpu_id}] | Model: {model_path} | rollouts len={len(rollout_id_list)} "
            f"| loading model (TP=1, enable_thinking={enable_thinking})...",
            flush=True,
        )

        llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=1)

        try:
            tokenizer = llm.get_tokenizer()
            for stop_token in ["<|im_end|>", "<|endoftext|>"]:
                try:
                    if hasattr(tokenizer, "encode"):
                        encoded = tokenizer.encode(stop_token, add_special_tokens=False)
                        if encoded:
                            stop_token_ids.append(encoded[0])
                except Exception:
                    pass
        except Exception as e:
            tokenizer = None
            print(f"[GPU {gpu_id}] Warning: Could not get tokenizer: {e}", flush=True)

        from tqdm import tqdm as tqdm_
        for rollout_id in tqdm_(rollout_id_list, desc=f"GPU {gpu_id}", position=int(gpu_id), leave=True):
            sampling = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                stop_token_ids=stop_token_ids if stop_token_ids else None,
            )

            if tokenizer is None:
                raise RuntimeError("Tokenizer is required for apply_chat_template but could not be loaded.")

            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": s["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for s in samples
            ]

            outputs = llm.generate(formatted_prompts, sampling, use_tqdm=False)

            for sample, out in zip(samples, outputs):
                results.append({
                    "example_id": sample["example_id"],
                    "prompt":     sample["prompt"],
                    "answer":     sample["answer"],
                    "seed":       rollout_id,
                    "response":   out.outputs[0].text,
                })

    except Exception as e:
        print(f"[GPU {gpu_id}] Critical Error: {e}", flush=True)

    finally:
        print(f"[GPU {gpu_id}] Cleaning up resources...", flush=True)
        if llm is not None:
            del llm
        if destroy_model_parallel is not None:
            try: destroy_model_parallel()
            except Exception: pass
        if destroy_distributed_environment is not None:
            try: destroy_distributed_environment()
            except Exception: pass
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[GPU {gpu_id}] Cleanup done.", flush=True)

    return results


def run_generation(model_name: str, step: int, weight_dir: Path, enable_thinking: bool):
    gpu_workers = [str(g) for g in GPUS]
    num_workers = len(gpu_workers)

    print(f"GPU workers: {gpu_workers}")
    print(f"apply_chat_template enable_thinking={enable_thinking}")
    print(f"\n{'='*50}\nGenerating: model={model_name} step={step}\n  weight={weight_dir}\n{'='*50}")

    out_dir = Path(f"{OUT_DIR_NAME}/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        task_name = task["name"]
        task_path = task["path"]
        N         = task["N"]

        out_path = out_dir / f"{task_name.lower()}_t{TEMPERATURE}_p{TOP_P}_n{N}-MNT{MAX_TOKENS}_step{step}.jsonl"

        print(f"Starting generation for task: {task_name} (N={N}) step={step}")

        if not REPLACE and not APPEND and out_path.exists():
            print(f"Result file already exists at '{out_path}'. Skipping.")
            continue

        existing_results = []
        done_rollout_ids: set[int] = set()
        if APPEND and not REPLACE:
            src_path = None
            if out_path.exists():
                src_path = out_path
            else:
                candidates = sorted(
                    out_dir.glob(f"{task_name.lower()}_t{TEMPERATURE}_p{TOP_P}_n*-MNT{MAX_TOKENS}_step{step}.jsonl")
                )
                if candidates:
                    src_path = candidates[-1]
            if src_path is not None:
                print(f"APPEND mode: loading existing rollouts from '{src_path}'")
                with src_path.open() as f:
                    for line in f:
                        item = json.loads(line)
                        existing_results.append(item)
                        done_rollout_ids.add(item["seed"])
                print(f"APPEND mode: {len(done_rollout_ids)} rollouts already done.")

        rollout_ids = [r for r in range(N) if r not in done_rollout_ids]

        if not rollout_ids:
            print(f"All {N} rollouts already present. Writing consolidated file to '{out_path}'.")
            with out_path.open("w", encoding="utf-8") as f:
                for item in existing_results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            continue

        print(f"Generating {len(rollout_ids)} rollout(s) (rollout_id {rollout_ids[0]}..{rollout_ids[-1]}).")

        samples = load_samples(task_path)
        for sample in samples:
            sample["prompt"] = PROMPT_TEMPLATE.format(problem=sample["prompt"])

        if samples:
            print("Example prompt after formatting:")
            print(samples[0]["prompt"])

        rollout_chunks = split_rollout_ids(rollout_ids, num_workers)

        args_list = [
            (str(weight_dir), samples, rollout_chunks[i], gpu_workers[i], enable_thinking)
            for i in range(num_workers)
        ]

        ctx = multiprocessing.get_context("spawn")
        all_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            futures = [ex.submit(worker_process, tup) for tup in args_list]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"GPU workers ({task_name}, step={step})"):
                try:
                    all_results.extend(fut.result())
                except Exception as e:
                    print(f"A worker process failed: {e}")

        print(f"Total new generations collected for {task_name} step={step}: {len(all_results)}")

        combined = existing_results + all_results
        if combined:
            with out_path.open("w", encoding="utf-8") as f:
                for item in combined:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved {len(combined)} total results for {task_name} step={step} to '{out_path}'")
        else:
            print(f"No results collected for {task_name} step={step}.")


# =========================================================================== #
#                          Grading helpers                                    #
# =========================================================================== #

def get_diverse_score(sequences, n=4):
    distinct_ngrams = set()
    total_ngrams = 0
    for seq in sequences:
        tokens = seq.split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            distinct_ngrams.add(ngram)
            total_ngrams += 1
    return len(distinct_ngrams) / total_ngrams if total_ngrams > 0 else 0


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        raise ValueError(f"k={k} cannot exceed n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def process_jsonl_file(file_name):
    results = []
    with open(file_name) as f:
        for line in f:
            data = json.loads(line)
            idx = int(data["example_id"])
            while len(results) <= idx:
                results.append({"gt": None, "responses": []})
            results[idx]["gt"] = data["answer"]
            results[idx]["responses"].append(data["response"])
            results[idx]["question"] = data.get("prompt", "")
    return results


def parse_hyperparameters_from_filename(filename):
    match = re.search(
        r"_t(?P<temperature>[\d.]+)_p(?P<top_p>[\d.]+)_n(?P<n>\d+)-MNT(?P<max_tokens>\d+)(?:_step(?P<step>\d+))?",
        filename,
    )
    return match.groupdict() if match else {}


def grade_file(file_path, length_tokenizer, vllm_model, model_tokenizer, sampling_params, use_model_verifier):
    hyperparams = parse_hyperparameters_from_filename(file_path.name)
    if not hyperparams:
        print(f"Skipping file with unrecognized format: {file_path}")
        return None

    hyperparams["task_name"] = file_path.stem.split("_")[0]

    if "parquet" in str(file_path):
        df = pd.read_parquet(file_path)
        num_pred = len(df["responses"][0])
    else:
        df = process_jsonl_file(file_path)
        num_pred = len(df[0]["responses"])

    results = {
        "hyperparameters": hyperparams,
        "mean_score": 0, "distinct_4gram": 0, "best_score": 0,
        "solve_none": 0, "solve_all": 0, "avg_output_length": 0,
        "format_error_rollouts": 0, "pass_at_k": {},
    }

    diverse, response_lengths = [], []
    without_boxed = 0
    all_model_inputs, rule_based_scores = [], []

    for i in range(len(df)):
        if "jsonl" in str(file_path):
            responses = df[i]["responses"]
            gt        = df[i]["gt"]
            question  = df[i].get("question", "")
        else:
            responses = df["responses"][i]
            gt        = df["reward_model"][i]["ground_truth"]
            question  = df["reward_model"][i].get("question", "")

        responses_list = [str(r) for r in responses]
        response_lengths += [len(length_tokenizer.encode(r)) for r in responses_list]
        without_boxed   += sum("boxed" not in r for r in responses_list)

        for response in responses_list:
            rule_score = grade_answer_verl(response, gt)
            rule_based_scores.append(rule_score)
            if not rule_score and use_model_verifier:
                all_model_inputs.append(CV_PROMPT.format(question=question, gold_answer=gt, llm_response=response))

        diverse.append(get_diverse_score(responses_list))

    model_based_scores = []
    if all_model_inputs and use_model_verifier and vllm_model is not None:
        model_inputs = [
            model_tokenizer.apply_chat_template(
                [{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False
            )
            for t in all_model_inputs
        ]
        for output in vllm_model.generate(model_inputs, sampling_params):
            model_based_scores.append("A" == output.outputs[0].text.strip())

    model_idx, final_scores = 0, []
    for rule_score in rule_based_scores:
        if rule_score:
            final_scores.append(True)
        elif use_model_verifier:
            final_scores.append(model_based_scores[model_idx])
            model_idx += 1
        else:
            final_scores.append(False)

    if final_scores:
        avg_scores = [sum(final_scores[i:i+num_pred]) / num_pred for i in range(0, len(final_scores), num_pred)]
        best       = [max(final_scores[i:i+num_pred])            for i in range(0, len(final_scores), num_pred)]
    else:
        avg_scores, best = [0], [0]

    results["mean_score"]           = sum(avg_scores) / len(avg_scores)
    results["distinct_4gram"]       = sum(diverse) / len(diverse) if diverse else 0
    results["best_score"]           = sum(best) / len(best)
    results["solve_none"]           = sum(1 for s in avg_scores if s == 0)
    results["solve_all"]            = sum(1 for s in avg_scores if s == 1)
    results["avg_output_length"]    = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    results["format_error_rollouts"] = without_boxed

    per_problem_correct = [sum(final_scores[i:i+num_pred]) for i in range(0, len(final_scores), num_pred)]
    results["pass_at_k"] = {
        f"pass@{k}": sum(estimate_pass_at_k(num_pred, c, k) for c in per_problem_correct) / len(per_problem_correct)
        for k in K_VALUES if k <= num_pred
    }

    return results


_STEP_FROM_FNAME_RE = re.compile(r"_step(\d+)\.jsonl$")


def _step_from_filename(name: str):
    m = _STEP_FROM_FNAME_RE.search(name)
    return int(m.group(1)) if m else None


def run_grading(model_name: str, step: int, use_model_verifier: bool, delete_rollouts: bool):
    """Grade rollouts for a single (model_name, step) and merge into grading_results.json.

    Output JSON shape: {"step{step}": [task_result, ...], ...}
    Existing entries for other steps are preserved; the entry for this `step`
    is overwritten cleanly.
    """
    length_tokenizer = AutoTokenizer.from_pretrained(LENGTH_TOKENIZER_PATH, local_files_only=True)

    vllm_model, model_tokenizer, sampling_params = None, None, None
    if use_model_verifier:
        print("Loading CompassVerifier model...")
        model_tokenizer = AutoTokenizer.from_pretrained(VERIFIER_MODEL_PATH)
        vllm_model      = LLM(model=VERIFIER_MODEL_PATH, tensor_parallel_size=8)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    else:
        print("Model verifier disabled. Running in rule-based only mode.")

    eval_dir    = Path(f"{OUT_DIR_NAME}/{model_name}")
    output_file = eval_dir / "grading_results.json"

    if not eval_dir.exists():
        print(f"Directory {eval_dir} does not exist. Nothing to grade.")
        return

    results_by_step: dict[str, list] = {}
    if output_file.exists():
        try:
            with output_file.open() as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                results_by_step = {k: v for k, v in loaded.items() if isinstance(v, list)}
        except Exception as e:
            print(f"Could not load existing {output_file}: {e}. Starting fresh.")

    # Clear this step so we overwrite cleanly instead of accumulating duplicates.
    results_by_step.pop(f"step{step}", None)

    graded_files: list[Path] = []
    for file_path in sorted(eval_dir.glob("*.jsonl")):
        file_step = _step_from_filename(file_path.name)
        if file_step is None:
            print(f"Skipping file without _step suffix: {file_path}")
            continue
        if file_step != step:
            continue
        print(f"Processing file: {file_path}")
        result = grade_file(file_path, length_tokenizer, vllm_model, model_tokenizer, sampling_params, use_model_verifier)
        if result:
            results_by_step.setdefault(f"step{step}", []).append(result)
            graded_files.append(file_path)

    ordered = {
        k: results_by_step[k]
        for k in sorted(results_by_step.keys(), key=lambda s: int(s[len("step"):]))
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=4)
    print(f"Grading results saved to {output_file}")

    if delete_rollouts:
        for fp in graded_files:
            try:
                fp.unlink()
                print(f"Deleted rollout file: {fp}")
            except Exception as e:
                print(f"Failed to delete {fp}: {e}")


# =========================================================================== #
#                                   main                                      #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_dir", type=str, default=WEIGHT_DIR)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    weight_dir = Path(args.weight_dir)
    model_name = args.model_name
    if not weight_dir.exists():
        print(f"WEIGHT_DIR does not exist: {weight_dir}")
        return

    print(f"\nTarget checkpoint:")
    print(f"  - model_name = {model_name}")
    print(f"  - step       = {STEP}")
    print(f"  - weight_dir = {weight_dir}")
    print(f"  - output_dir = {OUT_DIR_NAME}/{model_name}")

    if not SKIP_GEN:
        print("\n" + "="*60)
        print("  STEP 1 / 2 : Generation")
        print("="*60)
        run_generation(model_name, STEP, weight_dir, enable_thinking=ENABLE_THINKING)

    if not SKIP_GRADE:
        print("\n" + "="*60)
        print("  STEP 2 / 2 : Grading")
        print("="*60)
        run_grading(
            model_name,
            STEP,
            use_model_verifier=ENABLE_MODEL_VERIFIER,
            delete_rollouts=DELETE_ROLLOUTS,
        )


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
