#!/bin/bash
# MATH500 完整测试: v10 W8A8 多卡并行
#
# 用法:
#   bash scripts/run_math500_w8a8.sh          # 默认 16 卡
#   bash scripts/run_math500_w8a8.sh 8        # 指定卡数
#
# 结果:
#   results/math500_w8a8/math500_results.jsonl
#   results/math500_w8a8/math500_metrics.txt

set -eo pipefail

set +e
source /usr/local/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONUNBUFFERED=1

# ---- 配置 ----
NUM_DEVICES=${1:-16}
OM_MODEL_PATH="opt_models/v10_w8a8/om/DeepSeek-R1-Distill-Qwen-1.5B_32768_16.om"
RESULT_DIR="results/math500_w8a8"
# ---- 配置结束 ----

echo "=========================================="
echo " MATH500 Benchmark — v10 W8A8"
echo " OM:      ${OM_MODEL_PATH}"
echo " Devices: ${NUM_DEVICES}"
echo " Output:  ${RESULT_DIR}/"
echo "=========================================="

mkdir -p "$RESULT_DIR"

cat > /tmp/_math500_run.py << 'PYEOF'
import json, re, os, sys
from collections import defaultdict
import multiprocessing as mp

sys.path.insert(0, os.environ.get("SCRIPT_DIR", "."))

HF_MODEL_DIR = "/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
OM_MODEL_PATH = os.environ["OM_MODEL_PATH"]
DATASET_PATH = "/mnt/host-model/cxj/dataset/math500/test.jsonl"
RESULT_DIR = os.environ["RESULT_DIR"]
OUTPUT_PATH = os.path.join(RESULT_DIR, "math500_results.jsonl")
METRICS_TXT_PATH = os.path.join(RESULT_DIR, "math500_metrics.txt")

KV_CACHE_LENGTH = 32768
MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 31744
MAX_PREFILL_LENGTH = 16
K = 1
NUM_DEVICES = int(os.environ.get("NUM_DEVICES", "16"))


def normalize_answer(ans):
    ans = ans.strip()
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    ans = ans.replace(" ", "")
    return ans


def extract_boxed(text):
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else ""


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_finished_ids(output_path):
    finished = set()
    candidates = [output_path] + [os.path.join(RESULT_DIR, f"_tmp_device_{i}.jsonl") for i in range(NUM_DEVICES)]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    finished.add(json.loads(line)["unique_id"])
                except Exception:
                    pass
    return finished


def worker(device_id, items, result_path):
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
        sampling_method="top_p",
        sampling_value=0.95,
        temperature=0.6,
        system_prompt="",
        dtype="float16",
        device_str="npu",
    )
    engine = Inference(config)
    print(f"[device {device_id}] Model loaded, {len(items)} problems", flush=True)

    with open(result_path, "a", encoding="utf-8") as fout:
        for idx, item in enumerate(items):
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
            print(f"[device {device_id}] {idx+1}/{len(items)} "
                  f"{'✓' if correct else '✗'} "
                  f"pred={predicted!r:20s} gt={ground_truth!r:20s} "
                  f"{item['unique_id']}", flush=True)

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

    finished_ids = load_finished_ids(OUTPUT_PATH)
    if finished_ids:
        print(f"[resume] skipping {len(finished_ids)} finished items", flush=True)
    dataset = [item for item in dataset if item["unique_id"] not in finished_ids]
    if not dataset:
        print("All items already finished.", flush=True)
        return

    print(f"Running {len(dataset)} problems on {NUM_DEVICES} devices", flush=True)

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
            print(f"[WARN] device {i} exited with code {p.exitcode}", flush=True)

    # merge
    records = {}
    for path in tmp_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                records[r["unique_id"]] = r

    all_dataset = load_dataset(DATASET_PATH)
    ordered = [records[item["unique_id"]] for item in all_dataset if item["unique_id"] in records]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for r in ordered:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    # statistics
    correct_total = sum(r["pass1"] for r in ordered)
    subject_stats = defaultdict(lambda: [0.0, 0])
    level_stats = defaultdict(lambda: [0.0, 0])
    for r in ordered:
        correct_total_check = r["pass1"]
        subject_stats[r["subject"]][0] += r["pass1"]
        subject_stats[r["subject"]][1] += 1
        level_stats[r["level"]][0] += r["pass1"]
        level_stats[r["level"]][1] += 1

    total = len(ordered)
    lines = []
    lines.append(f"=== pass@1 (k={K}): {correct_total/total:.4f}  ({correct_total:.1f}/{total}) ===")
    lines.append("\n--- By Subject ---")
    for subj, (c, t) in sorted(subject_stats.items()):
        lines.append(f"  {subj}: {c/t:.4f}  ({c:.1f}/{t})")
    lines.append("\n--- By Level ---")
    for lvl, (c, t) in sorted(level_stats.items()):
        lines.append(f"  Level {lvl}: {c/t:.4f}  ({c:.1f}/{t})")

    metrics_text = "\n".join(lines)
    print("\n" + metrics_text, flush=True)

    with open(METRICS_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(metrics_text + "\n")
    print(f"\n结果已保存至: {METRICS_TXT_PATH}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
PYEOF

export SCRIPT_DIR="$SCRIPT_DIR"
export OM_MODEL_PATH
export RESULT_DIR
export NUM_DEVICES

conda run --no-capture-output -n qwen_ascend_cann900 \
  python3 /tmp/_math500_run.py

echo ""
echo "=========================================="
echo "结果文件:"
ls -lh "$RESULT_DIR"/*.jsonl "$RESULT_DIR"/*.txt 2>/dev/null
echo "=========================================="
