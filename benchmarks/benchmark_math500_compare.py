"""
MATH500 精度对比测试: v5 FP16 vs v12 W8A8+QKV
用法:
  conda run -n qwen_ascend_cann900 python3 benchmarks/benchmark_math500_compare.py \
    --device_id 5 --num_problems 1 --max_new_tokens 4000
"""
import json, re, os, sys, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HF_MODEL_DIR = "/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_PATH = "/mnt/host-model/cxj/dataset/math500/test.jsonl"

MODELS = {
    "v5_fp16": "opt_models/v10_reproduce/math500_om/v5_fp16_32k_32.om",
    "v12_w8a8_qkv": "opt_models/v10_reproduce/math500_om/v12_w8a8_qkv_maskfix_32k_32.om",
}

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

def load_dataset(path, num_problems, seed=42):
    with open(path, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]
    import random
    rng = random.Random(seed)
    indices = list(range(len(all_items)))
    rng.shuffle(indices)
    selected = [all_items[i] for i in indices[:num_problems]]
    return selected

def load_finished(out_path):
    """加载已完成的结果，返回 {unique_id: record}"""
    finished = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    finished[r["unique_id"]] = r
                except Exception:
                    pass
    return finished


def run_model(model_label, om_path, items, device_id, max_new_tokens, kv_cache_length, max_prefill_length, out_path):
    from config import InferenceConfig
    from utils.inference import Inference

    # 断点续跑: 加载已完成的结果
    finished = load_finished(out_path)
    if finished:
        print(f"  [{model_label}] 已有 {len(finished)}/{len(items)} 条结果, 续跑剩余部分", flush=True)

    config = InferenceConfig(
        hf_model_dir=HF_MODEL_DIR,
        om_model_path=om_path,
        onnx_model_path="",
        session_type="acl",
        device_id=device_id,
        kv_cache_length=kv_cache_length,
        max_output_length=kv_cache_length,
        max_input_length=1024,
        max_prefill_length=max_prefill_length,
        sampling_method="top_p",
        sampling_value=0.95,
        temperature=0.6,
        system_prompt="",
        dtype="float16",
        device_str="npu",
    )
    print(f"\n  [{model_label}] Loading OM model...", flush=True)
    engine = Inference(config)
    print(f"  [{model_label}] Model loaded, starting inference\n", flush=True)

    all_results = list(finished.values())
    correct = sum(1 for r in all_results if r['correct'])
    t0 = time.time()

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, item in enumerate(items):
            uid = item.get("unique_id", "")
            if uid in finished:
                continue

            engine.reset()
            ground_truth = normalize_answer(item["answer"])
            t_item = time.time()

            raw_output = engine.predict(
                item["problem"],
                history=[],
                system_prompt="",
                max_new_tokens=max_new_tokens,
            )
            item_elapsed = time.time() - t_item
            predicted = normalize_answer(extract_boxed(raw_output))
            is_correct = predicted == ground_truth
            if is_correct:
                correct += 1
            output_len = len(raw_output)

            done = len(all_results) + 1
            acc_so_far = correct / done
            elapsed = time.time() - t0
            print(f"  [{model_label}] {done}/{len(items)} "
                  f"{'✓' if is_correct else '✗'} "
                  f"pred={predicted!r:20s} gt={ground_truth!r:20s} "
                  f"acc={acc_so_far:.0%} "
                  f"({item_elapsed:.0f}s, total {elapsed:.0f}s, ~{output_len} chars)", flush=True)

            record = {
                "unique_id": uid,
                "subject": item.get("subject", ""),
                "level": item.get("level", ""),
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": is_correct,
                "raw_output": raw_output,
            }
            all_results.append(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    engine.session.close()
    elapsed = time.time() - t0
    accuracy = correct / len(all_results) if all_results else 0
    return all_results, accuracy, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=5)
    parser.add_argument("--num_problems", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4000)
    parser.add_argument("--kv_cache_length", type=int, default=4096)
    parser.add_argument("--max_prefill_length", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="opt_models/v10_reproduce/math500_results")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="要测试的模型列表")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    items = load_dataset(DATASET_PATH, args.num_problems)
    print(f"Selected {len(items)} problems (seed=42)", flush=True)
    for it in items[:3]:
        print(f"  {it['unique_id']}: {it['problem'][:60]}...", flush=True)

    all_results = {}
    for model_label in args.models:
        om_path = MODELS[model_label]
        if not os.path.exists(om_path):
            print(f"[SKIP] {model_label}: {om_path} not found", flush=True)
            continue
        print(f"\n{'='*60}", flush=True)
        print(f"Testing: {model_label} ({om_path})", flush=True)
        print(f"{'='*60}", flush=True)
        out_path = os.path.join(args.output_dir, f"{model_label}_results.jsonl")
        results, accuracy, elapsed = run_model(
            model_label, om_path, items, args.device_id,
            args.max_new_tokens, args.kv_cache_length, args.max_prefill_length, out_path)
        all_results[model_label] = (results, accuracy, elapsed)

    # Write summary to file
    summary_path = os.path.join(args.output_dir, "summary.txt")
    lines = []
    lines.append(f"MATH500 精度对比 ({len(items)} problems, seed=42)")
    lines.append(f"kv_cache={args.kv_cache_length}, prefill={args.max_prefill_length}, max_new_tokens={args.max_new_tokens}")
    lines.append("")
    lines.append(f"{'Model':<20} {'Accuracy':>10} {'Correct':>10} {'Total Time':>12}")
    lines.append("-" * 56)
    for label, (results, acc, elapsed) in all_results.items():
        c = sum(1 for r in results if r['correct'])
        lines.append(f"{label:<20} {acc:10.1%} {c:>7}/{len(results):<3} {elapsed:10.0f}s")

    if len(all_results) == 2:
        labels = list(all_results.keys())
        r0 = all_results[labels[0]][0]
        r1 = all_results[labels[1]][0]
        agree = sum(1 for a,b in zip(r0,r1) if a['correct'] == b['correct'])
        only_0 = sum(1 for a,b in zip(r0,r1) if a['correct'] and not b['correct'])
        only_1 = sum(1 for a,b in zip(r0,r1) if not a['correct'] and b['correct'])
        lines.append("")
        lines.append(f"一致: {agree}/{len(r0)}  仅{labels[0]}正确: {only_0}  仅{labels[1]}正确: {only_1}")

        lines.append("")
        lines.append("逐题对比:")
        lines.append(f"{'unique_id':<40} {labels[0]:>10} {labels[1]:>10} {'gt':>20}")
        lines.append("-" * 84)
        for a, b in zip(r0, r1):
            mark_a = "✓" if a['correct'] else "✗"
            mark_b = "✓" if b['correct'] else "✗"
            diff = "  ←" if a['correct'] != b['correct'] else ""
            lines.append(f"{a['unique_id']:<40} {mark_a:>10} {mark_b:>10} {a['ground_truth']:>20}{diff}")

    summary_text = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\n{'='*60}", flush=True)
    print(summary_text, flush=True)
    print(f"\n结果已保存至: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
