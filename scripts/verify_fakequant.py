"""
量化精度验证脚本：对比 FP16 原始 ONNX 和 fake_quant ONNX 的推理精度。

使用推理引擎的 KVCacheManager 构造输入（完整 prompt prefill），确保和实际推理一致。

验证内容:
  1. Prefill logits cosine similarity
  2. Top-5 token 匹配度
  3. 完整生成文本对比（引擎 decode loop）

用法:
  # 基础用法
  python3 scripts/verify_fakequant.py \
    --fp16_onnx opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --fakequant_onnx opt_models/v5_quant_w8a8/amct_output_v4/model_deploy_fake_quant_model.onnx \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B

  # 对比多个版本
  python3 scripts/verify_fakequant.py \
    --fp16_onnx opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --fakequant_onnx \
      v3=opt_models/v5_quant_w8a8/amct_output_v3/model_deploy_fake_quant_model.onnx \
      v4=opt_models/v5_quant_w8a8/amct_output_v4/model_deploy_fake_quant_model.onnx \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B
"""

import os
import sys
import argparse
import time
import numpy as np

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import amct_onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import InferenceConfig
from utils.kvcache import create_kv_cache
from transformers import AutoTokenizer


def cosine(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def make_session(onnx_path, register_amct=False, cpu_threads=64):
    opts = ort.SessionOptions()
    if register_amct:
        lib = os.path.join(os.path.dirname(amct_onnx.__file__), 'custom_op', 'libamct_onnx_ops.so')
        opts.register_custom_ops_library(lib)
    opts.intra_op_num_threads = cpu_threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, opts, providers=['CPUExecutionProvider'])


def make_kv_config(onnx_path, hf_model_dir, kv_cache_length=4096):
    return InferenceConfig(
        hf_model_dir=hf_model_dir, om_model_path='dummy.om', onnx_model_path=onnx_path,
        session_type='onnx', kv_cache_length=kv_cache_length, max_output_length=kv_cache_length,
        max_input_length=kv_cache_length - 1, max_prefill_length=1, cpu_thread=64,
    )


def prefill_logits(sess, onnx_path, hf_model_dir, tokenizer, prompt, kv_cache_length=4096):
    """完整 prompt prefill，返回最后一个 position 的 logits。"""
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors='np')['input_ids'].astype(np.int64).reshape(1, -1)
    config = make_kv_config(onnx_path, hf_model_dir, kv_cache_length)
    kv_mgr = create_kv_cache(config)
    cache, mask, pos_ids = kv_mgr.get_inputs(input_ids.shape[-1])
    result = sess.run(None, {
        'input_ids': input_ids, 'attention_mask': mask,
        'past_key_values': cache, 'position_ids': pos_ids,
    })
    return result[0][0, -1].astype(np.float32)


def generate(sess, onnx_path, hf_model_dir, tokenizer, prompt,
             max_tokens=60, kv_cache_length=4096):
    """完整推理：prefill + decode loop，用 KVCacheManager 管理 KV cache。"""
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors='np')['input_ids'].astype(np.int64).reshape(1, -1)
    config = make_kv_config(onnx_path, hf_model_dir, kv_cache_length)
    kv_mgr = create_kv_cache(config)
    ids_list = []
    cur = input_ids
    for _ in range(max_tokens):
        cache, mask, pos_ids = kv_mgr.get_inputs(cur.shape[-1])
        result = sess.run(None, {
            'input_ids': cur, 'attention_mask': mask,
            'past_key_values': cache, 'position_ids': pos_ids,
        })
        kv_mgr.update(cur.shape[-1], result[1])
        tok = int(np.argmax(result[0][0][-1:]))
        if tok == tokenizer.eos_token_id:
            break
        ids_list.append(tok)
        cur = np.array([[tok]], dtype=np.int64)
    return tokenizer.decode(ids_list, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="量化精度验证（使用推理引擎 KVCacheManager）")
    parser.add_argument("--fp16_onnx", type=str, required=True, help="FP16 原始 ONNX")
    parser.add_argument("--fakequant_onnx", nargs="+", required=True,
                        help="label=path 或 path（可多个）")
    parser.add_argument("--hf_model_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="请用一句话介绍量子计算。")
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--kv_cache_length", type=int, default=4096)
    parser.add_argument("--output", type=str, default="", help="结果保存路径")
    args = parser.parse_args()

    lines = []
    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)

    # 解析 fakequant 模型列表
    fq_models = {}
    for item in args.fakequant_onnx:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            label = os.path.basename(os.path.dirname(os.path.dirname(item)))
            path = item
        fq_models[label] = path

    # FP16 baseline
    log("=== FP16 baseline (prefill) ===")
    sess_fp16 = make_session(args.fp16_onnx, register_amct=False)
    logits_fp16 = prefill_logits(sess_fp16, args.fp16_onnx, args.hf_model_dir,
                                  tokenizer, args.prompt, args.kv_cache_length)
    top5_fp16 = np.argsort(logits_fp16)[::-1][:5]
    log(f"Top-5: {[tokenizer.decode([t]) for t in top5_fp16]}")

    log("\nFP16 generation:")
    gen_fp16 = generate(sess_fp16, args.fp16_onnx, args.hf_model_dir,
                        tokenizer, args.prompt, args.max_new_tokens, args.kv_cache_length)
    log(gen_fp16[:300])
    del sess_fp16

    # 各 FakeQuant 版本
    results = {}
    for label, path in fq_models.items():
        log(f"\n=== FakeQuant: {label} ===")
        sess_fq = make_session(path, register_amct=True)

        logits_fq = prefill_logits(sess_fq, path, args.hf_model_dir,
                                    tokenizer, args.prompt, args.kv_cache_length)
        top5_fq = np.argsort(logits_fq)[::-1][:5]

        cos = cosine(logits_fp16, logits_fq)
        max_diff = float(np.abs(logits_fp16 - logits_fq).max())
        top1_match = top5_fp16[0] == top5_fq[0]
        top5_overlap = len(set(top5_fp16) & set(top5_fq))

        log(f"Top-5: {[tokenizer.decode([t]) for t in top5_fq]}")
        log(f"Cosine vs FP16: {cos:.6f}")
        log(f"Max abs diff:   {max_diff:.4f}")
        log(f"Top-1 match:    {top1_match}")
        log(f"Top-5 overlap:  {top5_overlap}/5")

        log(f"\n{label} generation:")
        gen_fq = generate(sess_fq, path, args.hf_model_dir,
                          tokenizer, args.prompt, args.max_new_tokens, args.kv_cache_length)
        log(gen_fq[:300])

        if cos > 0.8:
            verdict = "PASS"
        elif cos > 0.6:
            verdict = "WARN (精度有偏移但可用)"
        else:
            verdict = "FAIL (精度崩溃)"
        log(f"Verdict: {verdict}")

        results[label] = {'cosine': cos, 'top1_match': top1_match,
                          'top5_overlap': top5_overlap, 'verdict': verdict}
        del sess_fq

    # Summary
    log(f"\n{'='*60}")
    log("Summary")
    log(f"{'='*60}")
    log(f"FP16 top-5: {[tokenizer.decode([t]) for t in top5_fp16]}")
    for label, r in results.items():
        log(f"  {label:20s}: cosine={r['cosine']:.4f}  top1={'Y' if r['top1_match'] else 'N'}  "
            f"top5={r['top5_overlap']}/5  {r['verdict']}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        log(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
