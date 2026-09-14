"""
ONNX 图改写：
  1. RoPE 融合: Slice+Neg+Concat+Mul+Mul+Add → NPURotaryPositionEmbedding
  2. QKV 合并: 3 个 q/k/v_proj MatMul → 1 个 qkv_proj MatMul + Split

用法:
  python export/change_node_v11_kv_inplace.py \
    --input_model_path output/onnx_xxx/model.onnx \
    --output_model_path output/onnx2_xxx/model_rectified.onnx
"""

import os
import onnx
import onnx.helper as helper
import numpy as np
from onnx import TensorProto, numpy_helper
from tqdm import tqdm
import argparse

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--input_model_path', type=str, required=True)
parser.add_argument('--output_model_path', type=str, required=True)
parser.add_argument('--skip_rope', action='store_true', default=False,
                    help="Skip RoPE fusion (for AMCT compatibility, keeps standard ONNX ops)")
parser.add_argument('--skip_qkv_merge', action='store_true', default=False,
                    help="Skip QKV merge pass")
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
if args.skip_rope:
    patterns = []
    print("[INFO] RoPE fusion skipped (--skip_rope)")
else:
    patterns = find_rope_patterns(model)
print(f"[INFO] Found {len(patterns)} RoPE patterns")

# ============================================================
# Pass 2: QKV MatMul merge
# Pattern per layer:
#   hidden ──┬── Transpose(W_q) ── MatMul ── Add(b_q) ── Q_out
#            ├── Transpose(W_k) ── MatMul ── Add(b_k) ── K_out
#            └── Transpose(W_v) ── MatMul ── Add(b_v) ── V_out
# Merge into:
#   hidden ── MatMul(W_qkv^T) ── Add(b_qkv) ── Split ── Q_out, K_out, V_out
# ============================================================

init_by_name = {init.name: init for init in model.graph.initializer}

def find_qkv_patterns(model):
    qkv_groups = []
    matmul_nodes = {}
    for node in model.graph.node:
        if node.op_type == 'MatMul':
            for proj in ['q_proj', 'k_proj', 'v_proj']:
                if f'/{proj}/MatMul' in node.name:
                    layer_prefix = node.name.split('/self_attn/')[0]
                    key = (layer_prefix, proj)
                    matmul_nodes[key] = node

    layer_prefixes = sorted(set(k[0] for k in matmul_nodes.keys()))
    for lp in layer_prefixes:
        q_mm = matmul_nodes.get((lp, 'q_proj'))
        k_mm = matmul_nodes.get((lp, 'k_proj'))
        v_mm = matmul_nodes.get((lp, 'v_proj'))
        if not (q_mm and k_mm and v_mm):
            continue
        if q_mm.input[0] != k_mm.input[0] or q_mm.input[0] != v_mm.input[0]:
            continue

        projs = {}
        valid = True
        for proj, mm in [('q', q_mm), ('k', k_mm), ('v', v_mm)]:
            tp = output_to_node.get(mm.input[1])
            if not tp or tp.op_type != 'Transpose':
                valid = False
                break
            w_init = init_by_name.get(tp.input[0])
            if w_init is None:
                valid = False
                break
            add_nodes = [n for n in model.graph.node
                         if n.op_type == 'Add' and mm.output[0] in n.input]
            if len(add_nodes) != 1:
                valid = False
                break
            add_node = add_nodes[0]
            bias_name = [i for i in add_node.input if i != mm.output[0]]
            if not bias_name:
                valid = False
                break
            b_init = init_by_name.get(bias_name[0])
            if b_init is None:
                valid = False
                break
            projs[proj] = {
                'matmul': mm, 'transpose': tp,
                'add': add_node, 'weight': w_init, 'bias': b_init,
            }
        if not valid or len(projs) != 3:
            continue
        qkv_groups.append({
            'layer_prefix': lp,
            'hidden_input': q_mm.input[0],
            'projs': projs,
        })
    return qkv_groups

print("[INFO] Searching for QKV merge patterns...")
if args.skip_qkv_merge:
    qkv_groups = []
    print("[INFO] QKV merge skipped (--skip_qkv_merge)")
else:
    qkv_groups = find_qkv_patterns(model)
print(f"[INFO] Found {len(qkv_groups)} QKV groups to merge")

qkv_nodes_to_remove = set()
qkv_new_nodes = {}
new_initializers = []

for group in qkv_groups:
    lp = group['layer_prefix']
    hidden_input = group['hidden_input']
    projs = group['projs']

    w_q = numpy_helper.to_array(projs['q']['weight'])
    w_k = numpy_helper.to_array(projs['k']['weight'])
    w_v = numpy_helper.to_array(projs['v']['weight'])
    b_q = numpy_helper.to_array(projs['q']['bias'])
    b_k = numpy_helper.to_array(projs['k']['bias'])
    b_v = numpy_helper.to_array(projs['v']['bias'])

    q_dim, k_dim, v_dim = w_q.shape[0], w_k.shape[0], w_v.shape[0]

    w_qkv = np.concatenate([w_q, w_k, w_v], axis=0)
    b_qkv = np.concatenate([b_q, b_k, b_v], axis=0)

    w_qkv_name = f"{lp}/self_attn/qkv_proj.weight"
    b_qkv_name = f"{lp}/self_attn/qkv_proj.bias"
    w_qkv_t = w_qkv.T.copy()

    new_initializers.append(numpy_helper.from_array(w_qkv_t, name=w_qkv_name))
    new_initializers.append(numpy_helper.from_array(b_qkv, name=b_qkv_name))

    mm_out = f"{lp}/self_attn/qkv_proj/MatMul_output_0"
    add_out = f"{lp}/self_attn/qkv_proj/Add_output_0"

    mm_node = helper.make_node(
        "MatMul", name=f"{lp}/self_attn/qkv_proj/MatMul",
        inputs=[hidden_input, w_qkv_name],
        outputs=[mm_out],
    )
    add_node = helper.make_node(
        "Add", name=f"{lp}/self_attn/qkv_proj/Add",
        inputs=[mm_out, b_qkv_name],
        outputs=[add_out],
    )

    q_out = projs['q']['add'].output[0]
    k_out = projs['k']['add'].output[0]
    v_out = projs['v']['add'].output[0]

    split_sizes_name = f"{lp}/self_attn/qkv_proj/split_sizes"
    new_initializers.append(numpy_helper.from_array(
        np.array([q_dim, k_dim, v_dim], dtype=np.int64), name=split_sizes_name
    ))

    split_node = helper.make_node(
        "Split", name=f"{lp}/self_attn/qkv_proj/Split",
        inputs=[add_out, split_sizes_name],
        outputs=[q_out, k_out, v_out],
        axis=-1,
    )

    insert_before_id = id(projs['q']['transpose'])
    qkv_new_nodes[insert_before_id] = [mm_node, add_node, split_node]

    for proj in projs.values():
        qkv_nodes_to_remove.add(id(proj['transpose']))
        qkv_nodes_to_remove.add(id(proj['matmul']))
        qkv_nodes_to_remove.add(id(proj['add']))

print(f"[INFO] QKV merge: removing {len(qkv_nodes_to_remove)} nodes, inserting {len(qkv_groups)*3} nodes")

if len(patterns) == 0:
    print("[WARN] No RoPE patterns found, falling back to standard change_node")
    new_nodes = []
    for node in tqdm(model.graph.node, desc="processing nodes"):
        if id(node) in qkv_nodes_to_remove:
            if id(node) in qkv_new_nodes:
                new_nodes.extend(qkv_new_nodes[id(node)])
            continue
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
        if id(node) in qkv_nodes_to_remove:
            if id(node) in qkv_new_nodes:
                new_nodes.extend(qkv_new_nodes[id(node)])
            continue

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

all_initializers = list(model.graph.initializer) + new_initializers

new_graph = helper.make_graph(
    new_nodes, "new_graph",
    inputs=model.graph.input,
    outputs=model.graph.output,
    value_info=model.graph.value_info,
    initializer=all_initializers
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
