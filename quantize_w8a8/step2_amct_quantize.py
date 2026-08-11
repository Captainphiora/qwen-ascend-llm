"""
AMCT ONNX Quantization - Produces real INT8 operators for Ascend NPU.
Unlike msmodelslim (which only quantizes weights), AMCT inserts AscendQuant/AscendDequant
nodes in the ONNX graph that ATC compiles into fused INT8 kernels.

Usage:
    python step2_amct_quantize.py \
        --onnx_model_path ./output/onnx_baseline/baseline_4096.onnx \
        --output_dir ./output/amct_w8a8 \
        --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
        --calib_file ./quantize_w8a8/calib_data/boolq.jsonl \
        --batch_num 10 \
        --kv_cache_length 4096
"""

import os
import sys
import json
import argparse
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_calib_texts(calib_file, num_samples=32):
    texts = []
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= num_samples:
                break
            data = json.loads(line.strip())
            if "inputs_pretokenized" in data:
                texts.append(data["inputs_pretokenized"])
            else:
                texts.append(str(list(data.values())[-1]))
    return texts


def main():
    parser = argparse.ArgumentParser(description="AMCT ONNX INT8 Quantization for Ascend")
    parser.add_argument("--onnx_model_path", type=str, required=True,
                        help="Input FP16 ONNX model (from export_onnx.py + change_node.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for quantized model")
    parser.add_argument("--hf_model_dir", type=str, required=True,
                        help="HuggingFace model dir (for tokenizer + config)")
    parser.add_argument("--calib_file", type=str, required=True,
                        help="Calibration data JSONL")
    parser.add_argument("--batch_num", type=int, default=10,
                        help="Number of calibration batches")
    parser.add_argument("--kv_cache_length", type=int, default=4096,
                        help="KV cache length (must match ONNX model)")
    parser.add_argument("--skip_layers", type=str, nargs="*", default=None,
                        help="Layers to skip quantization")
    args = parser.parse_args()

    import amct_onnx
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from transformers.models.qwen2 import Qwen2Config

    os.makedirs(args.output_dir, exist_ok=True)

    model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
    num_hidden_layers = model_config.num_hidden_layers
    num_key_value_heads = model_config.num_key_value_heads
    hidden_size = model_config.hidden_size
    num_attention_heads = model_config.num_attention_heads
    per_head_dim = hidden_size // num_attention_heads
    kv_cache_length = args.kv_cache_length

    config_file = os.path.join(args.output_dir, "quant_config.json")
    modified_onnx = os.path.join(args.output_dir, "model_modified.onnx")
    record_file = os.path.join(args.output_dir, "record.txt")
    save_path = os.path.join(args.output_dir, "model_deploy")

    print(f"[INFO] Creating quantization config...")
    amct_onnx.create_quant_config(
        config_file=config_file,
        model_file=args.onnx_model_path,
        skip_layers=args.skip_layers,
        batch_num=args.batch_num,
        activation_offset=True,
    )

    print(f"[INFO] Modifying model for calibration...")
    amct_onnx.quantize_model(
        config_file=config_file,
        model_file=args.onnx_model_path,
        modified_onnx_file=modified_onnx,
        record_file=record_file,
    )

    print(f"[INFO] Loading tokenizer for calibration data...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
    calib_texts = load_calib_texts(args.calib_file, num_samples=args.batch_num)
    print(f"[INFO] Loaded {len(calib_texts)} calibration texts")

    print(f"[INFO] Running calibration inference ({len(calib_texts)} batches)...")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    amct_so = os.path.join(
        os.path.dirname(amct_onnx.__file__),
        "custom_op", "libamct_onnx_ops.so"
    )
    if os.path.exists(amct_so):
        options.register_custom_ops_library(amct_so)
        print(f"[INFO] Registered AMCT custom ops: {amct_so}")

    session = ort.InferenceSession(
        modified_onnx,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    kv_shape = (1, kv_cache_length, num_hidden_layers * 2 * num_key_value_heads, per_head_dim)

    forward_count = 0
    for i, text in enumerate(calib_texts):
        encoded = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        input_ids = encoded["input_ids"].astype(np.int64)
        seq_len = input_ids.shape[1]

        token = input_ids[:, -1:]
        attention_mask = np.zeros((1, 1 + kv_cache_length), dtype=np.int64)
        attention_mask[0, -(seq_len):] = 1
        position_ids = np.array([[seq_len - 1]], dtype=np.int64)
        past_kv = np.zeros(kv_shape, dtype=np.float16)

        session.run(None, {
            "input_ids": token,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_kv,
        })
        forward_count += 1
        print(f"  Sample {i+1}/{len(calib_texts)} done (seq_len={seq_len})", flush=True)

    print(f"[INFO] Total calibration forward passes: {forward_count}", flush=True)

    print(f"[INFO] Saving quantized (deploy) model...")
    amct_onnx.save_model(
        modified_onnx_file=modified_onnx,
        record_file=record_file,
        save_path=save_path,
    )

    deploy_model = save_path + "_deploy_model.onnx"
    if os.path.exists(deploy_model):
        size_mb = os.path.getsize(deploy_model) / (1024 * 1024)
        print(f"\n[DONE] Quantized deploy model: {deploy_model} ({size_mb:.1f} MB)")
    else:
        for f in os.listdir(args.output_dir):
            print(f"  {f}")

    print(f"[INFO] Next step: compile with ATC using --precision_mode origin")


if __name__ == "__main__":
    main()
