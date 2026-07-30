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
        if node.op_type == "Cast":
            to_attribute = next(attr for attr in node.attribute if attr.name == "to")
            if to_attribute.i == TensorProto.INT8:
                new_node = helper.make_node(
                    "AscendQuant",
                    inputs=node.input,
                    outputs=node.output,
                    offset=0.,
                    scale=1.,
                )
        new_nodes.append(new_node)
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
