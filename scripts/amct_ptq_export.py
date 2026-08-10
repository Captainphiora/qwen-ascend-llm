"""
使用 amct_pytorch 对 FP16 模型做 W8A8 PTQ 量化，然后导出量化 ONNX。
使用与官方 msmodelslim 相同的 BoolQ 校准数据集。

用法:
  python3 scripts/amct_ptq_export.py \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --calib_file /usr/local/Ascend/atb-models/examples/convert/model_slim/boolq.jsonl \
    --output_dir output/amct_ptq \
    --kv_cache_length 4096 \
    --num_samples 16 \
    --device npu:0
"""

import os
import sys
import json
import argparse
import numpy as np

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_dir, "export"))


def load_boolq_prompts(calib_file: str, num_samples: int = 50):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_dir", type=str, required=True)
    parser.add_argument("--calib_file", type=str,
                        default="/usr/local/Ascend/atb-models/examples/convert/model_slim/boolq.jsonl")
    parser.add_argument("--output_dir", type=str, default="output/amct_ptq")
    parser.add_argument("--kv_cache_length", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="npu:0")
    args = parser.parse_args()

    device_str = args.device
    if "npu" in device_str:
        import torch_npu

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoTokenizer
    from transformers.models.qwen2 import Qwen2Config
    from modeling_qwen2 import Qwen2ForCausalLM

    print(f"[INFO] Loading model from: {args.hf_model_dir}")
    config = Qwen2Config.from_pretrained(args.hf_model_dir)
    model = Qwen2ForCausalLM.from_pretrained(
        args.hf_model_dir, torch_dtype=torch.float16
    ).to(device_str).eval()

    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    per_head_dim = config.hidden_size // num_attention_heads
    kv_dim = num_hidden_layers * 2 * num_key_value_heads

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)
    device = torch.device(device_str)

    print(f"[INFO] Loading calibration data from: {args.calib_file}")
    prompts = load_boolq_prompts(args.calib_file, args.num_samples)
    print(f"[INFO] Loaded {len(prompts)} calibration prompts")

    import amct_pytorch
    quant_config = amct_pytorch.INT8_SMOOTHQUANT_CFG.copy()
    quant_config["batch_num"] = len(prompts)
    quant_config["skip_layers"] = {"lm_head"}
    print(f"[INFO] Quantization config: {quant_config}")

    print("[INFO] Applying AMCT quantize...")
    amct_pytorch.quantize(model, config=quant_config)

    print("[INFO] Running calibration forward passes...")
    kv_cache_length = args.kv_cache_length

    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            tokens = tokenizer.encode(prompt)
            prompt_len = min(len(tokens), kv_cache_length - 1)

            input_ids = torch.tensor([[tokens[prompt_len - 1]]], dtype=torch.long).to(device)
            attention_mask = torch.ones((1, 1 + kv_cache_length), dtype=torch.long).to(device)
            attention_mask[:, prompt_len:kv_cache_length] = 0
            position_ids = torch.tensor([[prompt_len - 1]], dtype=torch.long).to(device)
            past_key_values = torch.randn(
                1, kv_cache_length, kv_dim, per_head_dim,
                dtype=torch.float16, device=device
            )

            model(input_ids, attention_mask, position_ids, past_key_values)

            if (idx + 1) % 10 == 0:
                print(f"  Calibration: {idx + 1}/{len(prompts)}")

    print("[INFO] Converting to deployment model...")
    amct_pytorch.convert(model)

    print("[INFO] Exporting quantized ONNX...")
    batch_size = 1
    seq_len = 1
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long).to(device)
    attention_mask = torch.zeros((batch_size, seq_len + kv_cache_length), dtype=torch.long).to(device)
    position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long).to(device)
    past_key_values = torch.randn(
        batch_size, kv_cache_length, kv_dim, per_head_dim,
        dtype=torch.float16, device=device
    )

    onnx_path = os.path.join(args.output_dir, "model_amct_int8.onnx")
    input_names = ["input_ids", "attention_mask", "position_ids", "past_key_values"]
    output_names = ["logits", "out_key_values"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length + kv_len"},
        "position_ids": {0: "batch_size", 1: "seq_length"},
        "past_key_values": {0: "batch_size", 1: "kv_len"},
    }

    torch.onnx.export(
        model,
        (input_ids, attention_mask, position_ids, past_key_values),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=False,
        opset_version=14,
        export_params=True,
    )
    print(f"[INFO] Quantized ONNX saved to: {onnx_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
