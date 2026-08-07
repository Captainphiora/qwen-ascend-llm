"""
生成atc calibration所需的校准数据 (方案B-2)
atc的--compression_optimize_conf支持calibration模式:
  将FP16模型的权重量化为INT8, 同时通过校准数据确定激活量化范围

用法:
  python export/quantize/gen_calibration_data.py \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output_dir output/calibration_data \
    --kv_cache_length 4096 \
    --num_samples 8
"""

import os
import sys
import argparse
import numpy as np

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"


project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CALIBRATION_PROMPTS = [
    "你好，请介绍一下你自己",
    "请详细介绍一下机器学习的基本概念和常用算法",
    "什么是深度学习？它和机器学习有什么区别？",
    "请解释一下Transformer架构的核心原理",
    "如何使用Python实现一个简单的排序算法？",
    "请介绍一下自然语言处理的发展历史",
    "什么是大语言模型？它有哪些应用场景？",
    "请解释一下注意力机制的工作原理",
]


def generate_calibration_bins(
    hf_model_dir: str,
    output_dir: str,
    kv_cache_length: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    per_head_dim: int,
    num_samples: int = 8,
    infer_soc: str = "Ascend910_9382",
):
    """
    生成atc calibration所需的bin文件
    atc要求每个输入一个bin文件, 多个sample的数据拼接在一起
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
    kv_dim = num_hidden_layers * 2 * num_key_value_heads

    os.makedirs(output_dir, exist_ok=True)

    input_ids_list = []
    attention_mask_list = []
    position_ids_list = []
    past_kv_list = []

    for idx in range(min(num_samples, len(CALIBRATION_PROMPTS))):
        prompt = CALIBRATION_PROMPTS[idx]
        tokens = tokenizer.encode(prompt)
        prompt_len = len(tokens)

        input_ids = np.array([[tokens[-1]]], dtype=np.int64)
        attention_mask = np.ones((1, 1 + kv_cache_length), dtype=np.int64)
        attention_mask[:, prompt_len:kv_cache_length] = 0
        position_ids = np.array([[prompt_len - 1]], dtype=np.int64)

        np.random.seed(idx)
        past_key_values = np.random.randn(
            1, kv_cache_length, kv_dim, per_head_dim
        ).astype(np.float16)

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
        past_kv_list.append(past_key_values)

    input_ids_bin = os.path.join(output_dir, "input_ids.bin")
    attention_mask_bin = os.path.join(output_dir, "attention_mask.bin")
    position_ids_bin = os.path.join(output_dir, "position_ids.bin")
    past_kv_bin = os.path.join(output_dir, "past_key_values.bin")

    np.concatenate(input_ids_list, axis=0).tofile(input_ids_bin)
    np.concatenate(attention_mask_list, axis=0).tofile(attention_mask_bin)
    np.concatenate(position_ids_list, axis=0).tofile(position_ids_bin)
    np.concatenate(past_kv_list, axis=0).tofile(past_kv_bin)

    print(f"[INFO] Generated calibration data ({num_samples} samples):")
    print(f"  {input_ids_bin} ({os.path.getsize(input_ids_bin)} bytes)")
    print(f"  {attention_mask_bin} ({os.path.getsize(attention_mask_bin)} bytes)")
    print(f"  {position_ids_bin} ({os.path.getsize(position_ids_bin)} bytes)")
    print(f"  {past_kv_bin} ({os.path.getsize(past_kv_bin) / 1024 / 1024:.1f} MB)")

    cfg_content = generate_compression_cfg(
        output_dir=output_dir,
        kv_cache_length=kv_cache_length,
        kv_dim=kv_dim,
        per_head_dim=per_head_dim,
        infer_soc=infer_soc,
    )
    cfg_path = os.path.join(output_dir, "compression_optimize.cfg")
    with open(cfg_path, "w") as f:
        f.write(cfg_content)
    print(f"\n[INFO] Compression config: {cfg_path}")
    print(f"  Use with: atc ... --compression_optimize_conf={cfg_path}")


def generate_compression_cfg(
    output_dir: str,
    kv_cache_length: int,
    kv_dim: int,
    per_head_dim: int,
    infer_soc: str = "Ascend910_9382",
) -> str:
    """
    生成atc --compression_optimize_conf 配置文件内容
    """
    abs_dir = os.path.abspath(output_dir)
    input_data_dir = ",".join([
        os.path.join(abs_dir, "input_ids.bin"),
        os.path.join(abs_dir, "attention_mask.bin"),
        os.path.join(abs_dir, "position_ids.bin"),
        os.path.join(abs_dir, "past_key_values.bin"),
    ])
    input_shape = (
        f"input_ids:1,1;"
        f"attention_mask:1,{1 + kv_cache_length};"
        f"position_ids:1,1;"
        f"past_key_values:1,{kv_cache_length},{kv_dim},{per_head_dim}"
    )

    cfg = f"""calibration:
{{
    input_data_dir: {input_data_dir}
    input_shape: {input_shape}
    infer_soc: {infer_soc}
    infer_device_id: 0
    log: info
}}
"""
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/calibration_data")
    parser.add_argument("--kv_cache_length", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--infer_soc", type=str, default="Ascend910_9382",
                        help="SoC name for calibration inference, e.g. Ascend910_9382")
    args = parser.parse_args()

    from transformers.models.qwen2 import Qwen2Config
    config = Qwen2Config.from_pretrained(args.hf_model_dir)
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    per_head_dim = config.hidden_size // config.num_attention_heads

    generate_calibration_bins(
        hf_model_dir=args.hf_model_dir,
        output_dir=args.output_dir,
        kv_cache_length=args.kv_cache_length,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        per_head_dim=per_head_dim,
        num_samples=args.num_samples,
        infer_soc=args.infer_soc,
    )


if __name__ == "__main__":
    main()
