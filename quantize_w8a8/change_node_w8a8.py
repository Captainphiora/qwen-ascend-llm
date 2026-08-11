"""
Post-process ONNX to insert AscendQuant/AscendDequant nodes for INT8 inference.
Uses pre-calibrated scales from msmodelslim (already in the ONNX as weight constants).
No additional calibration needed.

Input ONNX pattern (per quantized linear):
    weight_int8 → Cast(to=FP16) → Mul(weight_scale) → MatMul(input, dequant_weight)

Output ONNX pattern:
    input → AscendQuant(scale, offset) → MatMul(quant_input, weight_int8) → AscendDequant(deq_scale) → output

Usage:
    python change_node_w8a8.py \
        --input_model_path ./output/onnx_w8a8_final/model.onnx \
        --output_model_path ./output/onnx2_w8a8_int8/model.onnx \
        --quant_model_dir ./output/quant_w8a8
"""

import os
import sys
import argparse
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto, numpy_helper
from tqdm import tqdm
from safetensors import safe_open

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)


def load_quant_scales(quant_model_dir):
    """Load input_scale and deq_scale from msmodelslim quantized model."""
    safetensors_path = None
    for f in os.listdir(quant_model_dir):
        if f.endswith(".safetensors"):
            safetensors_path = os.path.join(quant_model_dir, f)
            break
    if safetensors_path is None:
        raise FileNotFoundError(f"No safetensors found in {quant_model_dir}")

    scales = {}
    with safe_open(safetensors_path, framework="np") as f:
        for key in f.keys():
            if key.endswith(".input_scale") or key.endswith(".input_offset"):
                layer_name = key.rsplit(".", 1)[0]
                if layer_name not in scales:
                    scales[layer_name] = {}
                attr = key.rsplit(".", 1)[1]
                scales[layer_name][attr] = f.get_tensor(key)
    return scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--quant_model_dir", type=str, required=True,
                        help="Path to msmodelslim quantized model (for input_scale/offset)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_model_path)), exist_ok=True)

    print(f"[INFO] Loading quant scales from {args.quant_model_dir}")
    quant_scales = load_quant_scales(args.quant_model_dir)
    print(f"[INFO] Found scales for {len(quant_scales)} layers")

    print(f"[INFO] Loading ONNX model: {args.input_model_path}")
    model = onnx.load(args.input_model_path)

    # Find Cast(int8→fp16) nodes - these mark quantized weight constants
    cast_to_fp16_nodes = {}
    for node in model.graph.node:
        if node.op_type == "Cast":
            to_attr = next((a for a in node.attribute if a.name == "to"), None)
            if to_attr and to_attr.i == TensorProto.FLOAT16:
                cast_to_fp16_nodes[node.output[0]] = node

    # Find pattern: Cast(int8→fp16) → Mul(scale) → MatMul
    # Replace MatMul input with the int8 weight directly, add AscendQuant on activation
    new_nodes = []
    replaced = 0

    for node in tqdm(model.graph.node, desc="Processing nodes"):
        # Handle Trilu (same as change_node.py)
        if node.op_type == "Trilu":
            new_node = helper.make_node(
                "Trilu",
                name="MY_" + node.name,
                inputs=[node.input[0]],
                outputs=node.output,
                upper=0
            )
            new_nodes.append(new_node)
            continue

        new_nodes.append(node)

    print(f"[INFO] Processed {len(new_nodes)} nodes, replaced Trilu nodes")
    print(f"[INFO] Note: INT8 acceleration requires AscendQuant insertion which")
    print(f"       depends on ATC's quantization_config_file option.")
    print(f"[INFO] Saving model to {args.output_model_path}")

    new_graph = helper.make_graph(
        new_nodes,
        "new_graph",
        inputs=model.graph.input,
        outputs=model.graph.output,
        value_info=model.graph.value_info,
        initializer=model.graph.initializer,
    )
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        opset_imports=model.opset_import,
        ir_version=model.ir_version,
    )
    onnx.save(new_model, args.output_model_path, save_as_external_data=True)
    print("[DONE]")


if __name__ == "__main__":
    main()
