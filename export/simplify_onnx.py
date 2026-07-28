"""Standalone script to simplify an existing ONNX model using onnxsim."""

import os
import argparse
from collections import Counter

import onnx
import onnxsim


def print_onnx_node_info(onnx_model_path: str):
    """Load an ONNX model and print node statistics."""
    model = onnx.load(onnx_model_path, load_external_data=False)
    nodes = model.graph.node
    total_nodes = len(nodes)
    op_counter = Counter(node.op_type for node in nodes)
    print("=" * 60)
    print(f"ONNX Node Statistics: {onnx_model_path}")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Unique op types: {len(op_counter)}")
    print("-" * 60)
    print(f"  {'Op Type':<30} {'Count':>6}")
    print("-" * 60)
    for op_type, count in op_counter.most_common():
        print(f"  {op_type:<30} {count:>6}")
    print("=" * 60)


def save_onnx_model(model, output_path: str):
    """Save ONNX model, using external data format if exceeding 2GB."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    data_file = os.path.basename(output_path) + ".data"
    onnx.save(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
    )


def simplify(input_path: str, output_path: str):
    """Simplify the ONNX model and save to output_path."""
    print(f"Loading model: {input_path}")
    input_dir = os.path.dirname(os.path.abspath(input_path))
    model = onnx.load(input_path, load_external_data=True)
    print("Original model:")
    print_onnx_node_info(input_path)

    print("\nRunning onnxsim...")
    model_sim, check = onnxsim.simplify(model)
    if not check:
        print("[ERROR] onnxsim simplification failed validation check")
        return

    save_onnx_model(model_sim, output_path)
    print(f"\nSimplified model saved to: {output_path}")
    print("\nSimplified model:")
    print_onnx_node_info(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify an ONNX model using onnxsim")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to the input ONNX model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="path to save the simplified model (default: <input>_sim.onnx)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        exit(1)

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_sim{ext}"

    simplify(args.input, args.output)
