"""
ONNX 图改写：将量化 W8A8 模式转换为 AscendQuant + QuantBatchMatmulV3

适用于 msmodelslim 量化模型 (quant_qwen.py --w_bit 8 --a_bit 8) 经
W8A8PreQuantizedLinear 导出的 ONNX。

输入图模式 (per quantized MatMul, from W8A8PreQuantizedLinear export):
  activation(FP16) ─────────────────────────────────── MatMul → output
  weight_int8 → Cast(FP16) → Mul(w_scale) → Transpose ─┘

替换为:
  activation(FP16) → AscendQuant(scale=input_scale) ─── QuantBatchMatmulV3 → output(FP16)
  weight_int8 (已转置存储) ──────────────────────────────┘
  w_scale (per-channel, float32) ────────────────────────┘

支持两种 scale 来源:
  1. --quant_model_path: 从 msmodelslim 量化的 safetensors 中读取每层校准的 input_scale
  2. --act_scale: 使用全局固定 scale (无校准数据, 精度较差, 仅用于兜底)

用法:
  # 使用校准的 per-layer input_scale (推荐)
  python export/quantize/change_node_quant.py \
    --input_model_path output/onnx_W8A8/model.onnx \
    --output_model_path output/onnx_quant_final/model.onnx \
    --quant_model_path /path/to/quantized/model.safetensors

  # 使用固定全局 scale (兜底)
  python export/quantize/change_node_quant.py \
    --input_model_path output/onnx_W8A8/model.onnx \
    --output_model_path output/onnx_quant_final/model.onnx \
    --act_scale 0.01
"""

import os
import argparse
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto, numpy_helper
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_model_path', type=str, required=True)
parser.add_argument('--output_model_path', type=str, required=True)
parser.add_argument('--act_scale', type=float, default=0.01,
                    help="Fallback global activation scale for AscendQuant")
parser.add_argument('--quant_model_path', type=str, default=None,
                    help="Path to msmodelslim quantized safetensors for per-layer input_scale")
args = parser.parse_args()

output_model_dir = os.path.dirname(os.path.abspath(args.output_model_path))
os.makedirs(output_model_dir, exist_ok=True)

per_layer_scales = {}
if args.quant_model_path and os.path.exists(args.quant_model_path):
    from safetensors import safe_open
    print(f"[INFO] Loading per-layer input_scale from: {args.quant_model_path}")
    with safe_open(args.quant_model_path, framework="numpy") as f:
        for key in f.keys():
            if key.endswith(".input_scale"):
                layer_prefix = key[: -len(".input_scale")]
                scale_val = f.get_tensor(key).astype(np.float32).item()
                per_layer_scales[layer_prefix] = scale_val
    print(f"[INFO] Loaded {len(per_layer_scales)} per-layer input_scale values")
else:
    print(f"[INFO] Using global act_scale={args.act_scale} (no quant_model_path provided)")

print(f"[INFO] Loading model: {args.input_model_path}")
model = onnx.load(args.input_model_path, load_external_data=True)

output_to_node = {}
for node in model.graph.node:
    for out in node.output:
        output_to_node[out] = node

init_map = {init.name: init for init in model.graph.initializer}

quant_patterns = []

for node in model.graph.node:
    if node.op_type != "MatMul":
        continue
    activation_input = node.input[0]
    weight_input = node.input[1]

    if weight_input not in output_to_node:
        continue
    transpose_node = output_to_node[weight_input]
    if transpose_node.op_type != "Transpose":
        continue

    mul_output = transpose_node.input[0]
    if mul_output not in output_to_node:
        continue
    mul_node = output_to_node[mul_output]
    if mul_node.op_type != "Mul":
        continue

    cast_output = mul_node.input[0]
    scale_name = mul_node.input[1]
    if cast_output not in output_to_node:
        continue
    cast_node = output_to_node[cast_output]
    if cast_node.op_type != "Cast":
        continue

    weight_int8_name = cast_node.input[0]
    if weight_int8_name not in init_map:
        continue
    weight_init = init_map[weight_int8_name]
    if weight_init.data_type != TensorProto.INT8:
        continue

    quant_patterns.append({
        "matmul_node": node,
        "transpose_node": transpose_node,
        "mul_node": mul_node,
        "cast_node": cast_node,
        "activation_input": activation_input,
        "weight_int8_name": weight_int8_name,
        "scale_name": scale_name,
        "matmul_output": node.output[0],
    })

print(f"[INFO] Found {len(quant_patterns)} quantized MatMul patterns to replace")


def get_layer_input_scale(weight_int8_name: str) -> float:
    """Derive per-layer input_scale from weight initializer name."""
    layer_prefix = weight_int8_name.replace(".weight_int8", "")
    if layer_prefix in per_layer_scales:
        return per_layer_scales[layer_prefix]
    if ".weight" in weight_int8_name:
        alt_prefix = weight_int8_name.replace(".weight", "")
        if alt_prefix in per_layer_scales:
            return per_layer_scales[alt_prefix]
    return args.act_scale


nodes_to_remove = set()
nodes_to_add = []
new_initializers = []

for idx, pattern in enumerate(quant_patterns):
    nodes_to_remove.add(id(pattern["matmul_node"]))
    nodes_to_remove.add(id(pattern["transpose_node"]))
    nodes_to_remove.add(id(pattern["mul_node"]))
    nodes_to_remove.add(id(pattern["cast_node"]))

    prefix = f"quant_{idx}"
    act_input = pattern["activation_input"]
    weight_name = pattern["weight_int8_name"]
    scale_name = pattern["scale_name"]
    matmul_output = pattern["matmul_output"]

    layer_scale = get_layer_input_scale(weight_name)

    quant_output = f"{prefix}_act_int8"
    ascend_quant_node = helper.make_node(
        "AscendQuant",
        inputs=[act_input],
        outputs=[quant_output],
        name=f"{prefix}_AscendQuant",
        scale=float(layer_scale),
        offset=0.0,
        dst_type=2,
    )
    nodes_to_add.append(ascend_quant_node)

    weight_init = init_map[weight_name]
    weight_np = numpy_helper.to_array(weight_init)
    weight_transposed = weight_np.T.copy()
    weight_t_name = f"{weight_name}_transposed"
    weight_t_tensor = numpy_helper.from_array(weight_transposed, name=weight_t_name)
    new_initializers.append(weight_t_tensor)

    scale_init = init_map[scale_name]
    scale_np = numpy_helper.to_array(scale_init).astype(np.float32).flatten()
    scale_f32_name = f"{scale_name}_f32"
    scale_f32_tensor = numpy_helper.from_array(scale_np, name=scale_f32_name)
    new_initializers.append(scale_f32_tensor)

    qbmm_node = helper.make_node(
        "QuantBatchMatmulV3",
        inputs=[quant_output, weight_t_name, scale_f32_name],
        outputs=[matmul_output],
        name=f"{prefix}_QuantBatchMatmulV3",
        dtype=1,
        transpose_x1=False,
        transpose_x2=True,
    )
    nodes_to_add.append(qbmm_node)

remaining_nodes = [n for n in model.graph.node if id(n) not in nodes_to_remove]
all_nodes = nodes_to_add + remaining_nodes
del model.graph.node[:]
model.graph.node.extend(all_nodes)

for init in new_initializers:
    model.graph.initializer.append(init)

print(f"[INFO] Replaced {len(quant_patterns)} patterns")
print(f"[INFO] Added {len(nodes_to_add)} new nodes")
print(f"[INFO] Total nodes: {len(model.graph.node)}")

if per_layer_scales:
    scales_used = [get_layer_input_scale(p["weight_int8_name"]) for p in quant_patterns]
    print(f"[INFO] input_scale range: [{min(scales_used):.6f}, {max(scales_used):.6f}]")

print(f"[INFO] Saving model: {args.output_model_path}")
data_file = os.path.basename(args.output_model_path) + ".data"
onnx.save(
    model,
    args.output_model_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=data_file,
)
print("[DONE]")
