"""
ONNX 图改写：将 RoPE 的 Slice+Neg+Concat+Mul+Mul+Add 模式
替换为昇腾 RotaryMul 融合算子。

RoPE 原始模式 (per Q/K per layer):
  Slice(x, :half) → x1
  Slice(x, half:) → x2
  Neg(x2) → -x2
  Concat(-x2, x1) → rotate_half(x)
  Mul(x, cos) → x*cos
  Mul(rotate_half(x), sin) → rotate_half(x)*sin
  Add(x*cos, rotate_half(x)*sin) → output

替换为:
  RotaryMul(x, cos, sin) → output

用法:
  python export/change_node_v1_rope.py \
    --input_model_path output/onnx_xxx/model.onnx \
    --output_model_path output/onnx2_xxx/model_rectified.onnx
"""

import os
import onnx
import onnx.helper as helper
from onnx import TensorProto
from tqdm import tqdm
import argparse

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--input_model_path', type=str, required=True)
parser.add_argument('--output_model_path', type=str, required=True)
args = parser.parse_args()

output_model_dir = os.path.dirname(os.path.abspath(args.output_model_path))
os.makedirs(output_model_dir, exist_ok=True)

print(f"[INFO] Loading model: {args.input_model_path}")
model = onnx.load(args.input_model_path)

# Build output→node index for pattern matching
output_to_node = {}
for node in model.graph.node:
    for o in node.output:
        output_to_node[o] = node

input_to_nodes = {}
for node in model.graph.node:
    for inp in node.input:
        if inp not in input_to_nodes:
            input_to_nodes[inp] = []
        input_to_nodes[inp].append(node)


def find_rope_patterns(model):
    """
    Find all RoPE patterns:
    Pattern: Add(Mul(x, cos), Mul(Concat(Neg(Slice(x)), Slice(x)), sin))
    
    We identify by: Add node whose two inputs are both Mul nodes,
    and one of the Mul's input comes from a Concat(Neg(Slice), Slice) chain.
    """
    patterns = []
    
    for node in model.graph.node:
        if node.op_type != 'Add':
            continue
        if 'self_attn/Add_1' not in node.name and 'self_attn/Add_2' not in node.name:
            continue
            
        add_node = node
        # Both inputs should come from Mul nodes
        mul_a_node = output_to_node.get(add_node.input[0])
        mul_b_node = output_to_node.get(add_node.input[1])
        
        if not mul_a_node or not mul_b_node:
            continue
        if mul_a_node.op_type != 'Mul' or mul_b_node.op_type != 'Mul':
            continue
        
        # Determine which Mul is x*cos and which is rotate_half(x)*sin
        # The one with Concat in its input chain is rotate_half(x)*sin
        x_cos_mul = None
        rot_sin_mul = None
        
        for mul_node in [mul_a_node, mul_b_node]:
            # Check if first input comes from Concat
            input0_node = output_to_node.get(mul_node.input[0])
            if input0_node and input0_node.op_type == 'Concat':
                rot_sin_mul = mul_node
            else:
                x_cos_mul = mul_node
        
        if not x_cos_mul or not rot_sin_mul:
            continue
        
        # x_cos_mul: Mul(x, cos) → x is input[0], cos is input[1]
        x_input = x_cos_mul.input[0]
        cos_input = x_cos_mul.input[1]
        sin_input = rot_sin_mul.input[1]
        output = add_node.output[0]
        
        # Collect all intermediate nodes to remove
        concat_node = output_to_node.get(rot_sin_mul.input[0])
        if not concat_node or concat_node.op_type != 'Concat':
            continue
            
        # Concat inputs: Neg output and Slice output
        intermediate_nodes = set()
        intermediate_nodes.add(id(add_node))
        intermediate_nodes.add(id(x_cos_mul))
        intermediate_nodes.add(id(rot_sin_mul))
        intermediate_nodes.add(id(concat_node))
        
        # Find Neg and Slices
        for inp in concat_node.input:
            n = output_to_node.get(inp)
            if n:
                intermediate_nodes.add(id(n))
                if n.op_type == 'Neg':
                    # Neg's input is from a Slice
                    slice_n = output_to_node.get(n.input[0])
                    if slice_n:
                        intermediate_nodes.add(id(slice_n))
        
        patterns.append({
            'x_input': x_input,
            'cos_input': cos_input,
            'sin_input': sin_input,
            'output': output,
            'nodes_to_remove': intermediate_nodes,
            'add_node_name': add_node.name,
        })
    
    return patterns


print("[INFO] Searching for RoPE patterns...")
patterns = find_rope_patterns(model)
print(f"[INFO] Found {len(patterns)} RoPE patterns")

if len(patterns) == 0:
    print("[WARN] No RoPE patterns found, falling back to standard change_node")
    # Do standard Trilu fix only
    new_nodes = []
    for node in tqdm(model.graph.node, desc="processing nodes"):
        new_node = node
        if node.op_type == "Trilu":
            new_node = helper.make_node(
                "Trilu", name="MY_" + node.name,
                inputs=[node.input[0]], outputs=node.output, upper=0
            )
        new_nodes.append(new_node)
else:
    # Collect all node IDs to remove
    all_nodes_to_remove = set()
    for p in patterns:
        all_nodes_to_remove.update(p['nodes_to_remove'])
    
    print(f"[INFO] Will remove {len(all_nodes_to_remove)} intermediate nodes")
    print(f"[INFO] Will insert {len(patterns)} NPURotaryPositionEmbedding nodes")
    
    new_nodes = []
    rotary_idx = 0
    
    for node in tqdm(model.graph.node, desc="replacing nodes"):
        if id(node) in all_nodes_to_remove:
            # Check if this is an Add node that should be replaced with RotaryMul
            for p in patterns:
                if id(node) == id(output_to_node.get(p['output'])):
                    # This is the Add node - replace with NPURotaryPositionEmbedding
                    rotary_node = helper.make_node(
                        "NPURotaryPositionEmbedding",
                        name=f"NPURotaryPosEmb_{rotary_idx}",
                        inputs=[p['x_input'], p['cos_input'], p['sin_input']],
                        outputs=[p['output']],
                        mode=0,  # 0="half" mode: split at D//2
                    )
                    new_nodes.append(rotary_node)
                    rotary_idx += 1
                    break
            # Skip other intermediate nodes (they're consumed by RotaryMul)
            continue
        
        # Standard change_node fixes
        new_node = node
        if node.op_type == "Trilu":
            new_node = helper.make_node(
                "Trilu", name="MY_" + node.name,
                inputs=[node.input[0]], outputs=node.output, upper=0
            )
        if node.op_type == "Cast":
            to_attribute = next((attr for attr in node.attribute if attr.name == "to"), None)
            if to_attribute and to_attribute.i == TensorProto.INT8:
                new_node = helper.make_node(
                    "AscendQuant",
                    inputs=node.input, outputs=node.output,
                    offset=0., scale=1.,
                )
        new_nodes.append(new_node)

print(f"[INFO] New graph: {len(new_nodes)} nodes (was {len(model.graph.node)})")

new_graph = helper.make_graph(
    new_nodes, "new_graph",
    inputs=model.graph.input,
    outputs=model.graph.output,
    value_info=model.graph.value_info,
    initializer=model.graph.initializer
)
new_model = helper.make_model(
    new_graph,
    producer_name=model.producer_name,
    opset_imports=model.opset_import,
    ir_version=model.ir_version
)

print(f"[INFO] Saving model: {args.output_model_path}")
onnx.save(new_model, args.output_model_path, save_as_external_data=True)
print("[DONE]")
