import os
import sys
from datetime import datetime
import onnx
import onnx.helper as helper
from onnx import TensorProto
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
    new_nodes = []

    # Collect INT8 initializer names (weight constants from quantization)
    int8_initializers = set()
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT8:
            int8_initializers.add(init.name)

    # Find Cast(INT8→FP16) nodes that take ONLY INT8 initializer as input (weight side only)
    weight_cast_outputs_to_bypass = {}  # cast_output → cast_input (the INT8 initializer)
    cast_to_int8_outputs = set()  # outputs of Cast(to=INT8) nodes

    for node in model.graph.node:
        if node.op_type == "Cast":
            to_attr = next((a for a in node.attribute if a.name == "to"), None)
            if to_attr:
                if to_attr.i == TensorProto.INT8:
                    cast_to_int8_outputs.add(node.output[0])
                elif to_attr.i == TensorProto.FLOAT16:
                    # ONLY bypass weight initializer casts, NOT activation casts
                    if node.input[0] in int8_initializers:
                        weight_cast_outputs_to_bypass[node.output[0]] = node.input[0]

    print(f"INT8 initializers: {len(int8_initializers)}")
    print(f"Cast(INT8→FP16) to bypass: {len(weight_cast_outputs_to_bypass)}")

    removed_casts = 0
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
            # Rewire MatMul inputs: bypass Cast(INT8→FP16), connect INT8 directly
            new_inputs = []
            rewired = False
            for inp in node.input:
                if inp in weight_cast_outputs_to_bypass:
                    new_inputs.append(weight_cast_outputs_to_bypass[inp])
                    rewired = True
                else:
                    new_inputs.append(inp)
            if rewired:
                new_node = helper.make_node(
                    "MatMul",
                    name=node.name,
                    inputs=new_inputs,
                    outputs=node.output,
                )
                removed_casts += 1
        new_nodes.append(new_node)

    print(f"Rewired {removed_casts} MatMul nodes to use INT8 inputs directly")
    print("make new graph")
    new_graph = helper.make_graph(
        new_nodes,
        "new_graph",
        inputs=model.graph.input,
        outputs=model.graph.output,
        value_info=model.graph.value_info,
        initializer=model.graph.initializer
    )
    print("make new model")
    new_model = helper.make_model(new_graph, producer_name=model.producer_name,opset_imports=model.opset_import,ir_version = model.ir_version)
    print("will save model in ", args.output_model_path)
    onnx.save(new_model, args.output_model_path, save_as_external_data=True)
finally:
    print(f"\n[LOG] Log saved to: {log_path}")
    sys.stdout = tee.terminal
    tee.close()
