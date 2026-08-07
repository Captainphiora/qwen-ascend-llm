"""
ONNX 模型 W8A16 权重量化脚本 (方案B-1)
直接将ONNX模型中的FP16 Linear权重量化为INT8, 插入DequantizeLinear节点
量化方式: per-channel symmetric INT8 (W8A16)
- 权重: INT8 (每个output channel一个scale)
- 激活: 保持FP16 (无精度损失)

原理:
  原始: X(fp16) @ W(fp16)^T
  量化后: X(fp16) @ DequantizeLinear(W_int8, scale)^T
  等价于: X(fp16) @ (W_int8 * scale)^T

用法:
  python export/quantize/quantize_weights.py \
    --input_model opt_models/v4_noexpand/onnx_changed/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --output_model output/onnx_quantized/DeepSeek-R1-Distill-Qwen-1.5B_4096_w8a16.onnx
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Tuple

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import onnx
from onnx import helper, TensorProto, numpy_helper


def find_linear_weights(model: onnx.ModelProto) -> List[Tuple[str, str]]:
    """
    找到所有 Linear 层的权重:
    模式: Initializer → Transpose → MatMul
    返回: [(initializer_name, transpose_node_name), ...]
    """
    initializer_names = {init.name for init in model.graph.initializer}
    transpose_to_init = {}

    for node in model.graph.node:
        if node.op_type == "Transpose" and node.input[0] in initializer_names:
            transpose_to_init[node.output[0]] = (node.input[0], node.name)

    linear_weights = []
    for node in model.graph.node:
        if node.op_type == "MatMul":
            if node.input[1] in transpose_to_init:
                init_name, transpose_name = transpose_to_init[node.input[1]]
                linear_weights.append((init_name, transpose_name, node.name))

    return linear_weights


def quantize_weight_per_channel(weight_fp16: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel symmetric INT8 量化
    Args:
        weight_fp16: FP16 权重 [out_features, in_features]
        axis: 量化轴 (0 = per output channel)
    Returns:
        weight_int8: INT8 权重
        scale: FP32 per-channel scale
    """
    weight_fp32 = weight_fp16.astype(np.float32)
    abs_max = np.abs(weight_fp32).max(axis=1, keepdims=True)
    abs_max = np.maximum(abs_max, 1e-8)
    scale = (abs_max / 127.0).squeeze(axis=1).astype(np.float32)
    weight_int8 = np.clip(
        np.round(weight_fp32 / abs_max * 127.0),
        -128, 127
    ).astype(np.int8)
    return weight_int8, scale


def quantize_onnx_weights(input_model: str, output_model: str, skip_lm_head: bool = True):
    """
    对ONNX模型执行W8A16量化
    Args:
        input_model: 输入ONNX模型路径
        output_model: 输出量化ONNX模型路径
        skip_lm_head: 是否跳过lm_head层(最后的输出投影)
    """
    print("=" * 60)
    print("[W8A16 Weight Quantization]")
    print(f"  Input:  {input_model}")
    print(f"  Output: {output_model}")
    print(f"  Skip lm_head: {skip_lm_head}")
    print("=" * 60)

    model = onnx.load(input_model, load_external_data=True)
    linear_weights = find_linear_weights(model)
    print(f"[INFO] Found {len(linear_weights)} linear layers to quantize")

    init_map = {init.name: init for init in model.graph.initializer}
    node_map = {node.name: node for node in model.graph.node}

    quantized_count = 0
    skipped_count = 0
    new_initializers = []
    nodes_to_remove = set()
    nodes_to_add = []

    for init_name, transpose_name, matmul_name in linear_weights:
        if skip_lm_head and "lm_head" in init_name:
            skipped_count += 1
            print(f"  [SKIP] {init_name} (lm_head)")
            continue

        init_tensor = init_map[init_name]
        weight_fp16 = numpy_helper.to_array(init_tensor)

        if weight_fp16.ndim != 2:
            skipped_count += 1
            print(f"  [SKIP] {init_name} (not 2D: shape={weight_fp16.shape})")
            continue

        weight_int8, scale = quantize_weight_per_channel(weight_fp16)

        int8_name = init_name + "_int8"
        scale_name = init_name + "_scale"
        zp_name = init_name + "_zero_point"
        dequant_output_name = init_name + "_dequantized"

        int8_tensor = numpy_helper.from_array(weight_int8, name=int8_name)
        scale_tensor = numpy_helper.from_array(scale, name=scale_name)
        zero_point = np.zeros(scale.shape, dtype=np.int8)
        zp_tensor = numpy_helper.from_array(zero_point, name=zp_name)

        new_initializers.extend([int8_tensor, scale_tensor, zp_tensor])

        dequant_node = helper.make_node(
            "DequantizeLinear",
            inputs=[int8_name, scale_name, zp_name],
            outputs=[dequant_output_name],
            name=f"DequantizeLinear_{init_name}",
            axis=0,
        )
        nodes_to_add.append(dequant_node)

        transpose_node = node_map[transpose_name]
        transpose_node_new_input = dequant_output_name
        for node in model.graph.node:
            if node.name == transpose_name:
                node.input[0] = dequant_output_name
                break

        quantized_count += 1

    for init_name, _, _ in linear_weights:
        if skip_lm_head and "lm_head" in init_name:
            continue
        init_tensor = init_map.get(init_name)
        if init_tensor is not None:
            model.graph.initializer.remove(init_tensor)

    for init in new_initializers:
        model.graph.initializer.append(init)

    existing_nodes = list(model.graph.node)
    all_nodes = nodes_to_add + existing_nodes
    del model.graph.node[:]
    model.graph.node.extend(all_nodes)

    print(f"\n[RESULT]")
    print(f"  Quantized layers: {quantized_count}")
    print(f"  Skipped layers:   {skipped_count}")
    print(f"  Total nodes:      {len(model.graph.node)}")

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    data_file = os.path.basename(output_model) + ".data"
    onnx.save(
        model,
        output_model,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
    )
    print(f"\n[INFO] Saved quantized model: {output_model}")

    orig_size = os.path.getsize(input_model)
    orig_data = input_model.replace(".onnx", "").split("/")
    model_dir = os.path.dirname(input_model)
    total_orig = sum(
        os.path.getsize(os.path.join(model_dir, f))
        for f in os.listdir(model_dir)
    )
    output_dir = os.path.dirname(output_model)
    total_quant = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.basename(output_model).replace(".onnx", "") in f
    )
    print(f"  Original total:  {total_orig / 1024 / 1024:.1f} MB")
    print(f"  Quantized total: {total_quant / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="W8A16 weight quantization for ONNX models")
    parser.add_argument("--input_model", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument(
        "--skip_lm_head",
        action="store_true",
        default=True,
        help="skip quantizing lm_head (output projection)",
    )
    parser.add_argument(
        "--no_skip_lm_head",
        action="store_false",
        dest="skip_lm_head",
    )
    args = parser.parse_args()
    quantize_onnx_weights(args.input_model, args.output_model, args.skip_lm_head)


if __name__ == "__main__":
    main()
