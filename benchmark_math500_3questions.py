import json
import re
import os
import multiprocessing as mp

HF_MODEL_DIR = "/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
OM_MODEL_PATH = "output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_32768_1_sim.om"
DATASET_PATH = "../dataset/math500/test.jsonl"
RESULT_DIR = "results_math500_3questions_greedy_test"
OUTPUT_PATH = os.path.join(RESULT_DIR, "math500_results.jsonl")
METRICS_TXT_PATH = os.path.join(RESULT_DIR, "math500_metrics.txt")
KV_CACHE_LENGTH = 32768
MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 31744
MAX_PREFILL_LENGTH = 1

TARGET_IDS = {
    "test/geometry/434.json",
    "test/intermediate_algebra/1197.json",
    "test/intermediate_algebra/1388.json",
}
NUM_DEVICES = 3


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
        max_output_length=KV_CACHE_LENGTH,
        max_input_length=MAX_INPUT_LENGTH,
        max_prefill_length=MAX_PREFILL_LENGTH,
        sampling_method="greedy",
        sampling_value=1.0,
        temperature=1.0,
        system_prompt="",
        dtype="float16",
        device_str="npu",
    )
    engine = Inference(config)

    with open(result_path, "a", encoding="utf-8") as fout:
        for item in items:
            ground_truth = normalize_answer(item["answer"])
            engine.reset()
            raw_output = engine.predict(
                item["problem"],
                history=[],
                system_prompt="",
                max_new_tokens=MAX_NEW_TOKENS,
            )
            predicted = normalize_answer(extract_boxed(raw_output))
            correct = predicted == ground_truth
            print(f"[device {device_id}] {item['unique_id']} correct={correct} pred={predicted!r} gt={ground_truth!r}")

            record = {
                "unique_id": item.get("unique_id", ""),
                "subject": item.get("subject", ""),
                "level": item.get("level", ""),
                "ground_truth": ground_truth,
                "pass1": 1.0 if correct else 0.0,
                "samples": [{"predicted_answer": predicted, "correct": correct, "raw_output": raw_output}],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    engine.session.close()


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    dataset = load_dataset(DATASET_PATH)
    dataset = [item for item in dataset if item["unique_id"] in TARGET_IDS]

    if not dataset:
        print("No matching items found in dataset.")
        return

    print(f"Found {len(dataset)} target items: {[d['unique_id'] for d in dataset]}")

    chunks = [dataset[i::NUM_DEVICES] for i in range(NUM_DEVICES)]
    tmp_paths = [os.path.join(RESULT_DIR, f"_tmp_device_{i}.jsonl") for i in range(NUM_DEVICES)]

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

    records = {}
    for path in tmp_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                records[r["unique_id"]] = r

    ordered = [records[item["unique_id"]] for item in dataset if item["unique_id"] in records]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for r in ordered:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    correct_total = sum(r["pass1"] for r in ordered)
    total = len(ordered)

    metrics_lines = []
    metrics_lines.append(f"=== greedy pass@1: {correct_total/total:.4f}  ({correct_total:.0f}/{total}) ===")
    metrics_lines.append(f"\n--- Per Question ---")
    for r in ordered:
        metrics_lines.append(f"  {r['unique_id']}: {'CORRECT' if r['pass1'] == 1.0 else 'WRONG'}  pred={r['samples'][0]['predicted_answer']!r}  gt={r['ground_truth']!r}")

    metrics_text = "\n".join(metrics_lines)
    print("\n" + metrics_text)

    with open(METRICS_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(metrics_text + "\n")
    print(f"\n[完成] 统计结果已保存至 {METRICS_TXT_PATH}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
