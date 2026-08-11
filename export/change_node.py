import os
import sys
from datetime import datetime
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto, numpy_helper
from tqdm import tqdm
import argparse


class TeeLogger:
    """Duplicate stdout to both console and a log file."""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
old_onnx_dir = os.path.join(output_dir, "onnx")
new_onnx_dir = os.path.join(output_dir, "onnx2")

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
model_name = "qwen2_1.5b_chat.onnx"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_model_path',
    type=str,
    help="raw onnx model convert by pytroch",
    default=os.path.join(old_onnx_dir, model_name)
)
parser.add_argument(
    "--output_model_path",
    help="output onnx model path",
    type=str,
    default=os.path.join(new_onnx_dir, model_name)
)
parser.add_argument(
    "--quant_model_dir",
    help="msmodelslim quantized model dir (for deq_scale). If not set, skip INT8 transform.",
    type=str,
    default="",
)

args = parser.parse_args()

log_dir = os.path.join(project_dir, "onnx_log")
input_basename = os.path.splitext(os.path.basename(args.input_model_path))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"change_node_{input_basename}_{timestamp}.log"
log_path = os.path.join(log_dir, log_filename)

tee = TeeLogger(log_path)
sys.stdout = tee

try:
    output_model_dir = os.path.dirname(os.path.abspath(args.output_model_path))
    os.makedirs(output_model_dir, exist_ok=True)
    for file in os.listdir(output_model_dir):
        file_path = os.path.join(output_model_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    model = onnx.load(args.input_model_path)

    # Load deq_scale from msmodelslim if provided
    deq_scales = {}
    if args.quant_model_dir:
        from safetensors import safe_open
        sf_path = None
        for f in os.listdir(args.quant_model_dir):
            if f.endswith(".safetensors"):
                sf_path = os.path.join(args.quant_model_dir, f)
                break
        if sf_path:
            with safe_open(sf_path, framework="np") as sf:
                for key in sf.keys():
                    if key.endswith(".deq_scale"):
                        layer = key.rsplit(".deq_scale", 1)[0]
                        # Store as UINT64 (hardware-native format)
                        deq_scales[layer] = sf.get_tensor(key).astype(np.uint64)
            print(f"Loaded UINT64 deq_scale for {len(deq_scales)} layers")

    # Collect INT8 initializer names
    int8_initializers = set()
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT8:
            int8_initializers.add(init.name)

    # Map: initializer_name → layer_name (e.g. "model.layers.0.mlp.gate_proj.weight_int8" → "model.layers.0.mlp.gate_proj")
    init_to_layer = {}
    for name in int8_initializers:
        if name.endswith(".weight_int8"):
            layer = name.rsplit(".weight_int8", 1)[0]
            init_to_layer[name] = layer

    # Find Cast(INT8→FP16) on weight side and activation side
    weight_cast_bypass = {}  # cast_output → INT8 initializer name
    act_cast_bypass = {}     # cast_output → AscendQuant/Cast(INT8) output name

    cast_to_int8_outputs = set()
    for node in model.graph.node:
        if node.op_type == "Cast":
            to_attr = next((a for a in node.attribute if a.name == "to"), None)
            if to_attr and to_attr.i == TensorProto.INT8:
                cast_to_int8_outputs.add(node.output[0])

    for node in model.graph.node:
        if node.op_type == "Cast":
            to_attr = next((a for a in node.attribute if a.name == "to"), None)
            if to_attr and to_attr.i == TensorProto.FLOAT16:
                if node.input[0] in int8_initializers:
                    weight_cast_bypass[node.output[0]] = node.input[0]
                elif node.input[0] in cast_to_int8_outputs:
                    act_cast_bypass[node.output[0]] = node.input[0]

    print(f"INT8 initializers: {len(int8_initializers)}")
    print(f"Weight Cast(INT8→FP16) to bypass: {len(weight_cast_bypass)}")
    print(f"Activation Cast(INT8→FP16) to bypass: {len(act_cast_bypass)}")

    # Track Transpose nodes that sit between Cast(INT8→FP16) and MatMul
    # Pattern: weight_int8 → Cast(FP16) → Transpose → MatMul
    transpose_bypass = {}  # transpose_output → (transpose_input_before_cast, weight_init_name)
    for node in model.graph.node:
        if node.op_type == "Transpose":
            if node.input[0] in weight_cast_bypass:
                transpose_bypass[node.output[0]] = weight_cast_bypass[node.input[0]]

    # Combined: MatMul input → INT8 source (through Cast and/or Transpose)
    matmul_weight_bypass = {}  # matmul_input_name → INT8 initializer name
    matmul_weight_bypass.update({k: v for k, v in weight_cast_bypass.items()})
    # For Transpose case, we need to keep the Transpose but change its input
    transpose_rewire = {}  # transpose_output → should rewire transpose input to INT8
    for node in model.graph.node:
        if node.op_type == "Transpose" and node.input[0] in weight_cast_bypass:
            transpose_rewire[node.output[0]] = (node, weight_cast_bypass[node.input[0]])

    print(f"Transpose nodes to rewire: {len(transpose_rewire)}")

    new_nodes = []
    new_initializers = list(model.graph.initializer)
    rewired_matmuls = 0
    inserted_dequants = 0

    for node in tqdm(model.graph.node, desc="replace node..."):
        new_node = node

        if node.op_type == "Trilu":
            new_node = helper.make_node(
                "Trilu",
                name="MY_" + node.name,
                inputs=[node.input[0]],
                outputs=node.output,
                upper=0
            )
        elif node.op_type == "Cast":
            to_attribute = next(attr for attr in node.attribute if attr.name == "to")
            if to_attribute.i == TensorProto.INT8:
                new_node = helper.make_node(
                    "AscendQuant",
                    inputs=node.input,
                    outputs=node.output,
                    offset=0.,
                    scale=1.,
                )
        elif node.op_type == "MatMul":
            # Check if this MatMul has INT8 inputs (via Cast bypass or Transpose bypass)
            new_inputs = list(node.input)
            weight_init_name = None
            has_int8_weight = False
            has_int8_act = False

            for idx, inp in enumerate(node.input):
                # Activation side: Cast(INT8→FP16) output directly feeds MatMul
                if inp in act_cast_bypass:
                    new_inputs[idx] = act_cast_bypass[inp]
                    has_int8_act = True
                # Weight side: Transpose output (Transpose takes Cast(INT8→FP16) output)
                if inp in transpose_rewire:
                    trans_node, init_name = transpose_rewire[inp]
                    weight_init_name = init_name
                    has_int8_weight = True
                    # We'll handle Transpose rewiring separately

            if has_int8_weight and has_int8_act and deq_scales:
                layer_name = init_to_layer.get(weight_init_name, "")
                deq_data = deq_scales.get(layer_name)

                if deq_data is not None:
                    matmul_int32_out = node.output[0] + "_int32"
                    deq_scale_name = f"{layer_name}.deq_scale_const"

                    # Use deq_scale as UINT64 (hardware-native Ascend format)
                    deq_init = numpy_helper.from_array(
                        deq_data, name=deq_scale_name
                    )
                    new_initializers.append(deq_init)

                    # MatMul with INT8 inputs
                    matmul_node = helper.make_node(
                        "MatMul", name=node.name,
                        inputs=new_inputs, outputs=[matmul_int32_out],
                    )
                    new_nodes.append(matmul_node)

                    # AscendDequant: INT32 × deq_scale → FP32 (then cast to FP16)
                    dequant_fp32_out = node.output[0] + "_fp32"
                    dequant_node = helper.make_node(
                        "AscendDequant",
                        inputs=[matmul_int32_out, deq_scale_name],
                        outputs=[dequant_fp32_out],
                    )
                    new_nodes.append(dequant_node)

                    # Cast FP32 → FP16 for downstream ops
                    cast_fp16_node = helper.make_node(
                        "Cast",
                        inputs=[dequant_fp32_out],
                        outputs=[node.output[0]],
                        to=TensorProto.FLOAT16,
                    )
                    new_nodes.append(cast_fp16_node)
                    rewired_matmuls += 1
                    inserted_dequants += 1
                    continue

        elif node.op_type == "Transpose":
            # Rewire Transpose input from Cast(FP16) output to INT8 initializer directly
            if node.output[0] in transpose_rewire:
                _, init_name = transpose_rewire[node.output[0]]
                new_node = helper.make_node(
                    "Transpose", name=node.name,
                    inputs=[init_name], outputs=node.output,
                    perm=[1, 0],
                )

        new_nodes.append(new_node)

    print(f"Rewired {rewired_matmuls} MatMul nodes")
    print(f"Inserted {inserted_dequants} AscendDequant nodes")
    print("make new graph")
    new_graph = helper.make_graph(
        new_nodes,
        "new_graph",
        inputs=model.graph.input,
        outputs=model.graph.output,
        value_info=model.graph.value_info,
        initializer=new_initializers
    )
    print("make new model")
    new_model = helper.make_model(new_graph, producer_name=model.producer_name,opset_imports=model.opset_import,ir_version = model.ir_version)
    print("will save model in ", args.output_model_path)
    onnx.save(new_model, args.output_model_path, save_as_external_data=True)
finally:
    print(f"\n[LOG] Log saved to: {log_path}")
    sys.stdout = tee.terminal
    tee.close()
