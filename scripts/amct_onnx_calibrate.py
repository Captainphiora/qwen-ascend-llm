"""
AMCT ONNX PTQ 量化校准脚本
使用 amct_onnx 对 FP16 ONNX 模型做 W8A8 校准量化，产出部署级量化 ONNX。

用法:
  python3 scripts/amct_onnx_calibrate.py \
    --model_path output/onnx_changed_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output_dir output/amct_onnx_ptq \
    --calib_file /usr/local/Ascend/atb-models/examples/convert/model_slim/boolq.jsonl \
    --num_samples 8 \
    --kv_cache_length 4096
"""

import os
import sys
import json
import time
import argparse
import numpy as np

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import amct_onnx
import onnxruntime as ort


def load_boolq_prompts(calib_file, num_samples):
    prompts = []
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            text = item.get("inputs_pretokenized", "")
            if text:
                prompts.append(text)
            if len(prompts) >= num_samples:
                break
    return prompts


def main():
    parser = argparse.ArgumentParser(description="AMCT ONNX PTQ calibration")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Input FP16 ONNX model path")
    parser.add_argument("--hf_model_dir", type=str, required=True,
                        help="HuggingFace model dir (for tokenizer and config)")
    parser.add_argument("--output_dir", type=str, default="output/amct_onnx_ptq")
    parser.add_argument("--calib_file", type=str,
                        default="/usr/local/Ascend/atb-models/examples/convert/model_slim/boolq.jsonl")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--kv_cache_length", type=int, default=4096)
    parser.add_argument("--cpu_threads", type=int, default=64,
                        help="Number of CPU threads for onnxruntime inference")
    parser.add_argument("--skip_layers", type=str, default="/lm_head/MatMul",
                        help="Comma-separated layer names to skip quantization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoTokenizer
    from transformers.models.qwen2 import Qwen2Config

    config = Qwen2Config.from_pretrained(args.hf_model_dir)
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    per_head_dim = config.hidden_size // num_attention_heads
    kv_dim = num_hidden_layers * 2 * num_key_value_heads

    print(f"[INFO] Model config: layers={num_hidden_layers}, kv_heads={num_key_value_heads}, "
          f"head_dim={per_head_dim}, kv_dim={kv_dim}")
    print(f"[INFO] ort version: {ort.__version__}, providers: {ort.get_available_providers()}")

    config_file = os.path.join(args.output_dir, "quant_config.json")
    modified_model = os.path.join(args.output_dir, "model_modified.onnx")
    record_file = os.path.join(args.output_dir, "record.txt")
    save_path = os.path.join(args.output_dir, "model_deploy")

    skip_layers = [s.strip() for s in args.skip_layers.split(",") if s.strip()]

    prompts = load_boolq_prompts(args.calib_file, args.num_samples)
    print(f"[INFO] Loaded {len(prompts)} calibration prompts from: {args.calib_file}")

    print("\n[Step 1] Creating quant config...")
    amct_onnx.create_quant_config(
        config_file=config_file,
        model_file=args.model_path,
        skip_layers=skip_layers,
        batch_num=len(prompts),
        activation_offset=True,
    )
    print(f"  Config: {config_file}")

    print("\n[Step 2] Inserting calibration nodes...")
    amct_onnx.quantize_model(
        config_file=config_file,
        model_file=args.model_path,
        modified_onnx_file=modified_model,
        record_file=record_file,
    )
    print(f"  Modified model: {modified_model}")

    print(f"\n[Step 3] Running calibration inference ({len(prompts)} samples, CPU)...")
    custom_op_lib = os.path.join(
        os.path.dirname(amct_onnx.__file__), "custom_op", "libamct_onnx_ops.so"
    )
    if not os.path.exists(custom_op_lib):
        print(f"[ERROR] Custom op lib not found: {custom_op_lib}")
        print("  Run: cd /tmp/amct_op_build/amct_onnx_op && python3 setup.py build")
        sys.exit(1)

    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(custom_op_lib)
    sess_options.intra_op_num_threads = args.cpu_threads
    sess_options.inter_op_num_threads = args.cpu_threads
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess = ort.InferenceSession(
        modified_model, sess_options, providers=["CPUExecutionProvider"]
    )
    print(f"  Session created (CPU, {args.cpu_threads} threads, parallel mode)")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)

    start_time = time.time()
    for idx, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        prompt_len = min(len(tokens), args.kv_cache_length - 1)

        input_ids = np.array([[tokens[prompt_len - 1]]], dtype=np.int64)
        attention_mask = np.ones((1, 1 + args.kv_cache_length), dtype=np.int64)
        attention_mask[:, prompt_len:args.kv_cache_length] = 0
        position_ids = np.array([[prompt_len - 1]], dtype=np.int64)
        past_key_values = np.zeros(
            (1, args.kv_cache_length, kv_dim, per_head_dim), dtype=np.float16
        )

        result = sess.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
        })
        elapsed = time.time() - start_time
        print(f"  Sample {idx+1}/{len(prompts)} done ({elapsed:.1f}s)")

    print(f"  Calibration completed in {time.time() - start_time:.1f}s")

    with open(record_file) as f:
        content = f.read()
    has_scale_d = "scale_d" in content
    print(f"  record has scale_d (activation scale): {has_scale_d}")

    if not has_scale_d:
        print("[ERROR] Missing scale_d! Activation calibration failed.")
        sys.exit(1)

    print("\n[Step 4] Saving deploy model...")
    amct_onnx.save_model(modified_model, record_file, save_path)
    deploy_onnx = f"{save_path}_deploy_model.onnx"
    print(f"  Deploy model: {deploy_onnx}")
    print("\n[DONE] Quantization complete!")


if __name__ == "__main__":
    main()
