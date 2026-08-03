import json
import re
import os
import sys
import math
from collections import defaultdict
from config import InferenceConfig
from utils.inference import Inference

HF_MODEL_DIR = "/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
OM_MODEL_PATH = "output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_65536_8.om"
DATASET_PATH = "../dataset/math500/test.jsonl"
OUTPUT_PATH = "results/math500_results.jsonl"
KV_CACHE_LENGTH = 65536
MAX_INPUT_LENGTH = 32768
MAX_NEW_TOKENS = 32768
MAX_PREFILL_LENGTH = 8
K = 4  # number of samples per question for pass@k


def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    ans = ans.replace(" ", "")
    return ans


def extract_boxed(text: str) -> str:
    # find all \boxed{...} and return the last one
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else ""


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    os.makedirs("results", exist_ok=True)

    config = InferenceConfig(
        hf_model_dir=HF_MODEL_DIR,
        om_model_path=OM_MODEL_PATH,
        onnx_model_path="",
        session_type="acl",
        kv_cache_length=KV_CACHE_LENGTH,
        max_output_length=KV_CACHE_LENGTH,
        max_input_length=MAX_INPUT_LENGTH,
        max_prefill_length=MAX_PREFILL_LENGTH,
        sampling_method="top_p",
        sampling_value=0.95,
        temperature=0.6,
        system_prompt="",
        dtype="float16",
        device_str="npu",
    )

    engine = Inference(config)
    dataset = load_dataset(DATASET_PATH)

    correct_total = 0
    subject_stats = defaultdict(lambda: [0, 0])  # [correct, total]
    level_stats = defaultdict(lambda: [0, 0])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for i, item in enumerate(dataset):
            problem = item["problem"]
            ground_truth = normalize_answer(item["answer"])
            subject = item.get("subject", "")
            level = item.get("level", "")
            unique_id = item.get("unique_id", "")

            samples = []
            for j in range(K):
                engine.reset()
                raw_output = engine.predict(
                    problem,
                    history=[],
                    system_prompt="",
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                predicted = normalize_answer(extract_boxed(raw_output))
                correct = predicted == ground_truth
                samples.append({"predicted_answer": predicted, "correct": correct, "raw_output": raw_output})
                print(f"[{i+1}/{len(dataset)}] sample {j+1}/{K} correct={correct} | pred={predicted!r} | gt={ground_truth!r}")

            pass1 = sum(s["correct"] for s in samples) / K
            correct_total += pass1
            subject_stats[subject][0] += pass1
            subject_stats[subject][1] += 1
            level_stats[level][0] += pass1
            level_stats[level][1] += 1

            record = {
                "unique_id": unique_id,
                "subject": subject,
                "level": level,
                "ground_truth": ground_truth,
                "pass1": pass1,
                "samples": samples,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    total = len(dataset)
    print(f"\n=== pass@1 (k={K}): {correct_total/total:.4f}  ({correct_total:.1f}/{total}) ===")
    print("\n--- By Subject ---")
    for subj, (c, t) in sorted(subject_stats.items()):
        print(f"  {subj}: {c/t:.4f}  ({c:.1f}/{t})")
    print("\n--- By Level ---")
    for lvl, (c, t) in sorted(level_stats.items()):
        print(f"  Level {lvl}: {c/t:.4f}  ({c:.1f}/{t})")


if __name__ == "__main__":
    main()
