import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import re
import os
from collections import defaultdict
import multiprocessing as mp


HF_MODEL_DIR = "../models/DeepSeek-R1-Distill-Qwen-1.5B"
OM_MODEL_PATH = "output/model/DeepSeek-R1-Distill-Qwen-1.5B_32768_1.om"
DATASET_PATH = "../dataset/math500/test.jsonl"
OUTPUT_PATH = "results/math500_results.jsonl"
# KV_CACHE_LENGTH = 32768
# MAX_INPUT_LENGTH = 32768
MAX_INPUT_LENGTH = 1024
KV_CACHE_LENGTH = 32768
MAX_NEW_TOKENS = 31744
MAX_PREFILL_LENGTH = 1
K = 1  # number of samples per question for pass@k
NUM_DEVICES = 1

def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    ans = ans.replace(" ", "")
    return ans


def extract_boxed(text: str) -> str:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else ""


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def worker(device_id: int, items: list, result_path: str):
    from config import InferenceConfig
    from utils.inference import Inference

    config = InferenceConfig(
        hf_model_dir=HF_MODEL_DIR,
        om_model_path=OM_MODEL_PATH,
        onnx_model_path="",
        session_type="acl",
        device_id=device_id,
        kv_cache_length=KV_CACHE_LENGTH,
        max_output_length=KV_CACHE_LENGTH-MAX_INPUT_LENGTH,
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

    with open(result_path, "a", encoding="utf-8") as fout:
        for item in items:
            ground_truth = normalize_answer(item["answer"])
            samples = []
            for j in range(K):
                engine.reset()
                raw_output = engine.predict(
                    item["problem"],
                    history=[],
                    system_prompt="",
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                predicted = normalize_answer(extract_boxed(raw_output))
                correct = predicted == ground_truth
                samples.append({"predicted_answer": predicted, "correct": correct, "raw_output": raw_output})
                print(f"[device {device_id}] {item['unique_id']} sample {j+1}/{K} correct={correct} pred={predicted!r}")

            pass1 = sum(s["correct"] for s in samples) / K
            record = {
                "unique_id": item.get("unique_id", ""),
                "subject": item.get("subject", ""),
                "level": item.get("level", ""),
                "ground_truth": ground_truth,
                "pass1": pass1,
                "samples": samples,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    engine.session.close()


def load_finished_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    finished = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                finished.add(json.loads(line)["unique_id"])
            except Exception:
                pass
    return finished


def main():
    os.makedirs("results", exist_ok=True)
    # dataset = load_dataset(DATASET_PATH)[:50]
    dataset = load_dataset(DATASET_PATH)

    finished_ids = load_finished_ids(OUTPUT_PATH)
    if finished_ids:
        print(f"[resume] skipping {len(finished_ids)} finished items")
    dataset = [item for item in dataset if item["unique_id"] not in finished_ids]
    if not dataset:
        print("All items already finished.")
        return

    # split dataset across devices
    chunks = [dataset[i::NUM_DEVICES] for i in range(NUM_DEVICES)]
    tmp_paths = [f"results/_tmp_device_{i}.jsonl" for i in range(NUM_DEVICES)]

    processes = []
    for i in range(NUM_DEVICES):
        if not chunks[i]:
            continue
        p = mp.Process(target=worker, args=(i, chunks[i], tmp_paths[i]))
        p.start()
        processes.append((i, p))

    for i, p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"[WARN] device {i} process exited with code {p.exitcode}")

    # merge results in original order, keyed by unique_id
    records = {}
    for path in tmp_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                records[r["unique_id"]] = r
        os.remove(path)

    ordered = [records[item["unique_id"]] for item in dataset if item["unique_id"] in records]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for r in ordered:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    # statistics
    correct_total = 0
    subject_stats = defaultdict(lambda: [0.0, 0])
    level_stats = defaultdict(lambda: [0.0, 0])
    for r in ordered:
        correct_total += r["pass1"]
        subject_stats[r["subject"]][0] += r["pass1"]
        subject_stats[r["subject"]][1] += 1
        level_stats[r["level"]][0] += r["pass1"]
        level_stats[r["level"]][1] += 1

    total = len(ordered)
    print(f"\n=== pass@1 (k={K}): {correct_total/total:.4f}  ({correct_total:.1f}/{total}) ===")
    print("\n--- By Subject ---")
    for subj, (c, t) in sorted(subject_stats.items()):
        print(f"  {subj}: {c/t:.4f}  ({c:.1f}/{t})")
    print("\n--- By Level ---")
    for lvl, (c, t) in sorted(level_stats.items()):
        print(f"  Level {lvl}: {c/t:.4f}  ({c:.1f}/{t})")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
