"""
量化精度验证（改进版）：对比 FP16 ONNX 和 fake_quant ONNX 的全 position logits。

改进点：
  1. 计算所有 position 的 logits cosine（不只是最后一个）
  2. 支持多 prompt 覆盖不同场景
  3. 分别报告 per-position cosine 的 min/mean/max

用法:
  conda run -n qwen_ascend_cann900 \
    python3 scripts/verify_fakequant_v2.py \
      --fp16_onnx opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
      --fakequant_onnx v8=opt_models/v10_reproduce/amct_output_v8_rerun/model_deploy_fake_quant_model.onnx \
      --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B
"""
import os, sys, argparse
import numpy as np

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import amct_onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import InferenceConfig
from utils.kvcache import create_kv_cache
from transformers import AutoTokenizer


def cosine(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def make_session(onnx_path, register_amct=False):
    opts = ort.SessionOptions()
    if register_amct:
        lib = os.path.join(os.path.dirname(amct_onnx.__file__), 'custom_op', 'libamct_onnx_ops.so')
        opts.register_custom_ops_library(lib)
    opts.intra_op_num_threads = 64
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(onnx_path, opts, providers=['CPUExecutionProvider'])


def prefill_all_logits(sess, onnx_path, hf_model_dir, tokenizer, prompt, kv_cache_length=4096):
    """完整 prompt prefill，返回所有 position 的 logits。"""
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors='np')['input_ids'].astype(np.int64).reshape(1, -1)
    cfg = InferenceConfig(
        hf_model_dir=hf_model_dir, om_model_path='dummy.om', onnx_model_path=onnx_path,
        session_type='onnx', kv_cache_length=kv_cache_length, max_output_length=kv_cache_length,
        max_input_length=kv_cache_length - 1, max_prefill_length=1, cpu_thread=64,
    )
    kv_mgr = create_kv_cache(cfg)
    cache, mask, pos_ids = kv_mgr.get_inputs(input_ids.shape[-1])
    result = sess.run(None, {
        'input_ids': input_ids, 'attention_mask': mask,
        'past_key_values': cache, 'position_ids': pos_ids,
    })
    return result[0][0].astype(np.float32)  # shape: (seq_len, vocab_size)


def analyze_logits(logits_fp16, logits_fq, tokenizer, label):
    """对比所有 position 的 logits，返回详细分析。"""
    seq_len = logits_fp16.shape[0]

    per_pos_cos = []
    for i in range(seq_len):
        per_pos_cos.append(cosine(logits_fp16[i], logits_fq[i]))

    cos_arr = np.array(per_pos_cos)
    last_cos = per_pos_cos[-1]
    all_cos = cosine(logits_fp16.flatten(), logits_fq.flatten())
    max_diff = float(np.abs(logits_fp16 - logits_fq).max())

    top5_fp16 = np.argsort(logits_fp16[-1])[::-1][:5]
    top5_fq = np.argsort(logits_fq[-1])[::-1][:5]
    top1_match = top5_fp16[0] == top5_fq[0]
    top5_overlap = len(set(top5_fp16) & set(top5_fq))

    print(f"\n=== {label} ({seq_len} positions) ===")
    print(f"  Per-position cosine:  min={cos_arr.min():.6f}  mean={cos_arr.mean():.6f}  max={cos_arr.max():.6f}")
    print(f"  Last-position cosine: {last_cos:.6f}  (旧脚本只算这个)")
    print(f"  All-position cosine:  {all_cos:.6f}  (所有 logits 展平后计算)")
    print(f"  Max abs diff:         {max_diff:.4f}")
    print(f"  Top-5 (FP16):  {[tokenizer.decode([t]) for t in top5_fp16]}")
    print(f"  Top-5 (quant): {[tokenizer.decode([t]) for t in top5_fq]}")
    print(f"  Top-1 match: {top1_match}  Top-5 overlap: {top5_overlap}/5")

    worst_pos = np.argmin(cos_arr)
    print(f"  Worst position: {worst_pos} (cosine={cos_arr[worst_pos]:.6f})")

    return {
        'per_pos_min': float(cos_arr.min()),
        'per_pos_mean': float(cos_arr.mean()),
        'last_cos': last_cos,
        'all_cos': all_cos,
        'max_diff': max_diff,
        'top1_match': top1_match,
        'top5_overlap': top5_overlap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16_onnx", required=True)
    parser.add_argument("--fakequant_onnx", nargs="+", required=True)
    parser.add_argument("--hf_model_dir", required=True)
    parser.add_argument("--kv_cache_length", type=int, default=4096)
    args = parser.parse_args()

    prompts = [
        ("中文推理", "请用一句话介绍量子计算。"),
        ("English QA", "What is the capital of France? Answer in one sentence."),
        ("代码", "Write a Python function to check if a number is prime."),
        ("数学", "Solve: 2x + 3 = 15. Show your steps."),
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)

    fq_models = {}
    for item in args.fakequant_onnx:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            label, path = "fq", item
        fq_models[label] = path

    sess_fp16 = make_session(args.fp16_onnx, register_amct=False)

    for fq_label, fq_path in fq_models.items():
        sess_fq = make_session(fq_path, register_amct=True)

        print(f"\n{'=' * 70}")
        print(f"FakeQuant: {fq_label}")
        print(f"{'=' * 70}")

        all_results = []
        for prompt_label, prompt_text in prompts:
            logits_fp16 = prefill_all_logits(
                sess_fp16, args.fp16_onnx, args.hf_model_dir,
                tokenizer, prompt_text, args.kv_cache_length)
            logits_fq = prefill_all_logits(
                sess_fq, fq_path, args.hf_model_dir,
                tokenizer, prompt_text, args.kv_cache_length)
            r = analyze_logits(logits_fp16, logits_fq, tokenizer, prompt_label)
            all_results.append((prompt_label, r))

        print(f"\n{'=' * 70}")
        print(f"Summary: {fq_label}")
        print(f"{'=' * 70}")
        print(f"{'Prompt':<12} {'last_cos':>10} {'all_cos':>10} {'pos_mean':>10} {'pos_min':>10} {'max_diff':>10}")
        for plabel, r in all_results:
            print(f"{plabel:<12} {r['last_cos']:10.6f} {r['all_cos']:10.6f} "
                  f"{r['per_pos_mean']:10.6f} {r['per_pos_min']:10.6f} {r['max_diff']:10.4f}")

        del sess_fq


if __name__ == "__main__":
    main()
