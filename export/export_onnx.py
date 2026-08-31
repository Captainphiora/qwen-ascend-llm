"""_summary_
qwen2 modeling_qwen2.py download: https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/qwen2/modeling_qwen2.py
"""

import os
import json
import sys
from typing import List
from datetime import datetime
import torch
import shutil
# from transformers import AutoModel, Qwen2Config
from transformers.models.qwen2 import Qwen2Config
from modeling_qwen2 import Qwen2ForCausalLM

import onnx
import io
import argparse
from collections import Counter

# Monkey-patch: fix safetensors files without metadata (e.g. from msmodelslim quantization)
import safetensors
_original_safe_open = safetensors.safe_open

class _SafeOpenWrapper:
    def __init__(self, *args, **kwargs):
        self._f = _original_safe_open(*args, **kwargs)

    def metadata(self):
        meta = self._f.metadata()
        if meta is None:
            return {"format": "pt"}
        return meta

    def __getattr__(self, name):
        return getattr(self._f, name)

    def __enter__(self):
        self._f.__enter__()
        return self

    def __exit__(self, *args):
        return self._f.__exit__(*args)

safetensors.safe_open = _SafeOpenWrapper
import transformers.modeling_utils as _tmu
if hasattr(_tmu, 'safe_open'):
    _tmu.safe_open = _SafeOpenWrapper


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
onnx_model_dir = os.path.join(output_dir, "onnx")


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device_str",
        type=str,
        choices=["npu", "cuda", "cpu"],
        help="support npu, cuda, cpu",
        default="cpu",
    )
    parser.add_argument(
        "--dtype" ,
        type=str,
        help="support float16/float32, if use CPU, only support fp32",
        choices=["float16", "float32"],
        default="float32",
    )
    parser.add_argument(
        '--hf_model_dir',
        type=str,
        help="model and tokenizer path, only support huggingface model",
        default=os.path.join(project_dir, "download", "Qwen2-1.5B-Instruct")
    )
    parser.add_argument(
        "--onnx_model_path",
        help="output onnx path",
        type=str,
        default=os.path.join(onnx_model_dir, "qwen2_1.5b_chat.onnx")
    )
    parser.add_argument(
        "--kv_cache_length",
        help="kv-cache length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--simplify",
        help="run onnxsim to simplify the exported ONNX model",
        type=str,
        choices=["true", "false"],
        default="false",
    )
    parser.add_argument(
        "--quantize",
        help="quantize mode: none, W8X8, W8A16, W8A8",
        type=str,
        choices=["none", "W8X8", "W8A16", "W8A8"],
        default="none",
    )
    parser.add_argument(
        "--kv_cache_layout",
        help="KV cache layout: BSHD (default) or BHSD (transpose-eliminated)",
        type=str,
        choices=["BSHD", "BHSD"],
        default="BSHD",
    )
    return parser.parse_args()


def print_onnx_node_info(onnx_model_path: str):
    """Load the exported ONNX model and print node statistics."""
    model = onnx.load(onnx_model_path)
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


def simplify_onnx(onnx_model_path: str, output_path: str = None):
    """Simplify the ONNX model using onnxsim and save it."""
    try:
        import onnxsim
    except ImportError:
        print("[ERROR] onnxsim not installed. Install via: pip install onnxsim")
        return None
    if output_path is None:
        base, ext = os.path.splitext(onnx_model_path)
        output_path = f"{base}_sim{ext}"
    print(f"Running onnxsim on: {onnx_model_path}")
    model = onnx.load(onnx_model_path, load_external_data=True)
    model_sim, check = onnxsim.simplify(model)
    if check:
        data_file = os.path.basename(output_path) + ".data"
        onnx.save(
            model_sim,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
        )
        print(f"Simplified model saved to: {output_path}")
        print_onnx_node_info(output_path)
        return output_path
    else:
        print("[WARNING] onnxsim simplification failed validation check")
        return None


def export_onnx(
    device_str,
    dtype: str,
    hf_model_dir: str,
    onnx_model_path: str,
    kv_cache_length: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    per_head_dim: int,
    quantize_mode: str = "none",
    kv_cache_layout: str = "BSHD",
):
    if device_str == "npu":
        import torch_npu
    if dtype == "float16":
        assert device_str.lower() != "cpu", print("cpu not support fp16")
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise Exception("unsupport dtype")

    device = torch.device(device_str)

    if quantize_mode == "W8A8":
        from quantize.w8a8_linear import load_w8a8_state_dict
        safetensors_path = os.path.join(hf_model_dir, "model.safetensors")
        if not os.path.exists(safetensors_path):
            candidates = [f for f in os.listdir(hf_model_dir) if f.endswith(".safetensors")]
            if candidates:
                safetensors_path = os.path.join(hf_model_dir, candidates[0])
            else:
                raise FileNotFoundError(f"No safetensors file found in {hf_model_dir}")
        print(f"[INFO] Loading W8A8 pre-quantized model from: {safetensors_path}")
        config = Qwen2Config.from_pretrained(hf_model_dir)
        model = Qwen2ForCausalLM(config).to(torch_dtype)
        model = load_w8a8_state_dict(
            safetensors_path, model, dtype=torch_dtype, device=device,
            skip_layers=["lm_head"]
        )
        model = model.to(device)
        print(f"[INFO] W8A8 model loaded successfully, {sum(1 for m in model.modules() if hasattr(m, 'weight_int8'))} quantized layers")
    else:
        model = Qwen2ForCausalLM.from_pretrained(
            hf_model_dir,
            torch_dtype=torch_dtype,
        ).to(device)
    quantize_cfg = {
        "q_proj": {
            "type": "W8X8",
            "act_scale": False
        },
        "k_proj": {
            "type": "W8X8",
            "act_scale": False
        },
        "v_proj": {
            "type": "W8X8",
            "act_scale": False
        },
        "o_proj": {
            "type": "W8X8",
            "act_scale": False
        },
        "gate_proj": {
            "type": "W8X8",
            "act_scale": False
        },
        "up_proj": {
            "type": "W8X8",
            "act_scale": False
        },
        "down_proj": {
            "type": "W8X8",
            "act_scale": False
        }
    }
    if quantize_mode in ("none", "W8A8"):
        quantize_cfg = {}
    elif quantize_mode == "W8A16":
        for key in quantize_cfg:
            quantize_cfg[key]["type"] = "W8A16"
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values"
    ]
    output_names = ["logits", "out_key_values"]
    # 四个输入中，哪些维度需要动态，对于past_key_values 后两维分别是2*层数*头数，hidden_dim，不需要动态
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length + kv_len"},
        "position_ids": {0: "batch_size", 1: "seq_length"},
        "past_key_values": {0: "batch_size", 1: "kv_len" if kv_cache_layout == "BSHD" else "num_heads"},
    }
    batch_size = 1
    seq_len = 1
    all_len = seq_len + kv_cache_length

    input_ids = torch.zeros((batch_size, seq_len)).long().to(device)
    attention_mask = torch.zeros((batch_size, all_len)).long().to(device)
    position_ids = torch.zeros((batch_size, seq_len)).long().to(device)
    if kv_cache_layout == "BHSD":
        past_key_values = torch.rand(
            (
                1,
                num_hidden_layers * 2 * num_key_value_heads,
                kv_cache_length,
                per_head_dim
            ),
            dtype=torch_dtype
        ).to(device)
    else:
        past_key_values = torch.rand(
            (
                1,
                kv_cache_length,
                num_hidden_layers * 2 * num_key_value_heads,
                per_head_dim
            ),
            dtype=torch_dtype
        ).to(device)
    input_args = (
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        # None,  # inputs_embeds: Optional[torch.FloatTensor] = None,
        # None,  # labels: Optional[torch.LongTensor] = None,
        # True,  # use_cache: Optional[bool] = None,
        # True,  # output_attentions: Optional[bool] = None,
        # None,  # output_hidden_states
        # False  # return_dict:
    )
    model.eval()
    with torch.no_grad():
        if quantize_cfg:
            from quantize.quantize_linear import quantize_model
            quantize_model(model, cfg=quantize_cfg)
            print(f"[INFO] Model quantized with mode: {quantize_mode}")
        torch.onnx.export(
            model,
            f=onnx_model_path,
            args=input_args,
            input_names=input_names,
            output_names=output_names,
            # 指定哪些维度是动态的
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            opset_version=14,
            export_params=True
        )


if __name__ == "__main__":
    args = parser_arguments()
    onnx_model_dir = os.path.dirname(os.path.abspath(args.onnx_model_path))
    os.makedirs(onnx_model_dir, exist_ok=True)

    log_dir = os.path.join(project_dir, "onnx_log")
    onnx_basename = os.path.splitext(os.path.basename(args.onnx_model_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"export_{onnx_basename}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    tee = TeeLogger(log_path)
    sys.stdout = tee

    try:
        if len(os.listdir(onnx_model_dir)) > 0:
            print("found some file in {}, will clear it".format(onnx_model_dir))
            for temp_file in os.listdir(onnx_model_dir):
                temp_path = os.path.join(onnx_model_dir, temp_file)
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
        src_file_path = os.path.join(now_dir, "modeling_qwen2.py")
        target_file_path = os.path.join(args.hf_model_dir, "modeling_qwen2.py")
        shutil.copy(src_file_path, target_file_path)
        config_json = os.path.join(args.hf_model_dir, "config.json")
        with open(config_json, "rt", encoding="utf-8") as f:
            model_config = json.load(f)
        model_config["auto_map"] = {
            "AutoModel": "modeling_qwen2.Qwen2ForCausalLM",
            "AutoModelForCausalLM": "modeling_qwen2.Qwen2ForCausalLM",
            "AutoModelForSeq2SeqLM": "modeling_qwen2.Qwen2ForCausalLM",
            "AutoModelForSequenceClassification": "modeling_qwen2.Qwen2ForSequenceClassification"
        }
        # Remove msmodelslim's custom quantization_config to avoid transformers
        # misinterpreting it as bitsandbytes config during from_pretrained
        model_config.pop("quantization_config", None)
        with open(config_json, "wt", encoding="utf-8") as f:
            json.dump(model_config, f, indent=4)
        test_model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
        test_model_config.torch_dtype = "float16"
        test_model_config.save_pretrained(args.hf_model_dir)
        num_hidden_layers = test_model_config.num_hidden_layers
        num_attention_heads = test_model_config.num_attention_heads
        num_key_value_heads = test_model_config.num_key_value_heads
        hidden_size = test_model_config.hidden_size
        per_head_dim = hidden_size // num_attention_heads
        print("new model config save ok in ", args.hf_model_dir)
        print("begin export onnx")
        export_onnx(
            device_str=args.device_str,
            dtype=args.dtype,
            hf_model_dir=args.hf_model_dir,
            onnx_model_path=args.onnx_model_path,
            kv_cache_length=args.kv_cache_length,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            per_head_dim=per_head_dim,
            quantize_mode=args.quantize,
            kv_cache_layout=args.kv_cache_layout,
        )
        print("onnx export done, save in ", args.onnx_model_path)
        print_onnx_node_info(args.onnx_model_path)
        if args.simplify == "true":
            simplify_onnx(args.onnx_model_path)
    finally:
        print(f"\n[LOG] Log saved to: {log_path}")
        sys.stdout = tee.terminal
        tee.close()
