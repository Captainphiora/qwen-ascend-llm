"""
ONNX 图改写：将量化 W8X8 模式转换为 AscendQuant + QuantBatchMatmulV3
用于方案C的量化模型编译

输入图模式 (per quantized MatMul):
  activation(FP16) ─────────────────────────────────── MatMul → output
  weight_int8 → Cast(FP16) → Mul(w_scale) → Transpose ─┘

替换为:
  activation(FP16) → AscendQuant(scale=act_scale) ─── QuantBatchMatmulV3 → output(FP16)
  weight_int8 (已转置存储) ─────────────────────────────┘
  w_scale (per-channel, float32) ───────────────────────┘

注意: AscendQuant 使用固定 scale (无校准数据), 精度会有损失

用法:
  python export/quantize/change_node_quant.py \
    --input_model_path output/onnx_changed_W8X8/model.onnx \
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
                    help="Static activation scale for AscendQuant (1/dynamic_range)")
args = parser.parse_args()

output_model_dir = os.path.dirname(os.path.abspath(args.output_model_path))
os.makedirs(output_model_dir, exist_ok=True)

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

    quant_output = f"{prefix}_act_int8"
    ascend_quant_node = helper.make_node(
        "AscendQuant",
        inputs=[act_input],
        outputs=[quant_output],
        name=f"{prefix}_AscendQuant",
        scale=float(args.act_scale),
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
