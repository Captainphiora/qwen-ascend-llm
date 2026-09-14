"""
校准脚本输入构造正确性验证:
用同一个 FP16 ONNX 模型，对比两种输入构造方式的 logits 输出:
  A) 校准脚本的方式 (amct_onnx_calibrate.py 的输入构造逻辑)
  B) 推理引擎的方式 (KVCacheManager.get_inputs)

如果两者 logits 完全一致 → 校准输入构造正确
如果不一致 → 校准输入有 bug，并可通过 diff 定位问题
"""
import os, sys
import numpy as np

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime as ort
from transformers import AutoTokenizer
from transformers.models.qwen2 import Qwen2Config
from config import InferenceConfig
from utils.kvcache import create_kv_cache


def make_session(onnx_path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 64
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, opts, providers=['CPUExecutionProvider'])


def calibration_style_inputs(tokens, config, kv_cache_length=4096):
    """校准脚本 (amct_onnx_calibrate.py) 的输入构造方式"""
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    per_head_dim = config.hidden_size // num_attention_heads
    kv_dim = num_hidden_layers * 2 * num_key_value_heads

    seq_len = len(tokens)
    input_ids = np.array([tokens], dtype=np.int64)
    attention_mask = np.ones((1, 1 + seq_len), dtype=np.int64)
    attention_mask[:, 0] = 0
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    past_key_values = np.zeros((1, 1, kv_dim, per_head_dim), dtype=np.float16)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }


def engine_style_inputs(tokens, onnx_path, hf_model_dir, kv_cache_length=4096):
    """推理引擎 (KVCacheManager) 的输入构造方式"""
    cfg = InferenceConfig(
        hf_model_dir=hf_model_dir,
        om_model_path='dummy.om',
        onnx_model_path=onnx_path,
        session_type='onnx',
        kv_cache_length=kv_cache_length,
        max_output_length=kv_cache_length,
        max_input_length=kv_cache_length - 1,
        max_prefill_length=1,
        cpu_thread=64,
    )
    kv_mgr = create_kv_cache(cfg)
    seq_len = len(tokens)
    cache, mask, pos_ids = kv_mgr.get_inputs(seq_len)

    return {
        "input_ids": np.array([tokens], dtype=np.int64),
        "attention_mask": mask,
        "position_ids": pos_ids,
        "past_key_values": cache,
    }


def compare_inputs(inputs_a, inputs_b, label_a="calibration", label_b="engine"):
    """逐字段对比两组输入"""
    all_match = True
    for key in inputs_a:
        a, b = inputs_a[key], inputs_b[key]
        if a.shape != b.shape:
            print(f"  {key}: SHAPE MISMATCH  {label_a}={a.shape}  {label_b}={b.shape}")
            all_match = False
        elif not np.array_equal(a, b):
            diff_positions = np.where(a != b)
            n_diff = len(diff_positions[0])
            print(f"  {key}: VALUE MISMATCH  ({n_diff} elements differ)")
            print(f"    {label_a}: {a.flatten()[:10]}...")
            print(f"    {label_b}: {b.flatten()[:10]}...")
            if key == "attention_mask":
                print(f"    {label_a} mask: ...{a[0, :5].tolist()}...{a[0, -5:].tolist()}")
                print(f"    {label_b} mask: ...{b[0, :5].tolist()}...{b[0, -5:].tolist()}")
            all_match = False
        else:
            print(f"  {key}: OK  shape={a.shape}")
    return all_match


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--hf_model_dir", required=True)
    parser.add_argument("--prompt", default="请用一句话介绍量子计算。")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)
    config = Qwen2Config.from_pretrained(args.hf_model_dir)

    messages = [{'role': 'user', 'content': args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(text)
    print(f"Prompt tokens: {len(tokens)}")

    print("\n[1] Constructing inputs...")
    inputs_calib = calibration_style_inputs(tokens, config)
    inputs_engine = engine_style_inputs(tokens, args.onnx_path, args.hf_model_dir)

    print("\n[2] Comparing inputs field-by-field:")
    inputs_match = compare_inputs(inputs_calib, inputs_engine)
    if inputs_match:
        print("\n  ALL INPUTS MATCH — calibration input construction is correct")
    else:
        print("\n  INPUTS DIFFER — running both through ONNX to compare outputs...")

    print("\n[3] Running ONNX inference with both input sets...")
    sess = make_session(args.onnx_path)

    result_calib = sess.run(None, inputs_calib)
    result_engine = sess.run(None, inputs_engine)

    logits_calib = result_calib[0][0, -1].astype(np.float32)
    logits_engine = result_engine[0][0, -1].astype(np.float32)

    cos = float(np.dot(logits_calib, logits_engine) /
                (np.linalg.norm(logits_calib) * np.linalg.norm(logits_engine)))
    max_diff = float(np.abs(logits_calib - logits_engine).max())
    top5_calib = np.argsort(logits_calib)[::-1][:5]
    top5_engine = np.argsort(logits_engine)[::-1][:5]

    print(f"\n[4] Logits comparison (last position):")
    print(f"  Cosine similarity: {cos:.10f}")
    print(f"  Max abs diff:      {max_diff:.8f}")
    print(f"  Top-5 (calibration): {[tokenizer.decode([t]) for t in top5_calib]}")
    print(f"  Top-5 (engine):      {[tokenizer.decode([t]) for t in top5_engine]}")
    print(f"  Top-5 match:         {set(top5_calib) == set(top5_engine)}")

    if cos > 0.999999 and max_diff < 0.01:
        print(f"\n  VERDICT: PASS — logits are identical, calibration inputs are correct")
    elif cos > 0.99:
        print(f"\n  VERDICT: WARN — logits are close but not identical (numerical noise?)")
    else:
        print(f"\n  VERDICT: FAIL — logits differ significantly, calibration inputs are WRONG")


if __name__ == "__main__":
    main()
