"""
W8A8 Pre-Quantized Linear layer for msmodelslim quantized models.

Loads int8 weights + calibrated scales from msmodelslim quantization output,
and produces an ONNX-friendly forward that creates recognizable patterns
for downstream graph rewriting (change_node_quant.py).

ONNX pattern produced per linear layer:
  weight_int8 → Cast(to=FP16) → Mul(weight_scale) ─┐
  activation(FP16) ────────────────────────────────── MatMul → Add(bias) → output


The input_scale / deq_scale / quant_bias are stored as named buffers so they
can be extracted by change_node scripts for Ascend operator conversion.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from safetensors import safe_open
import numpy as np


class W8A8PreQuantizedLinear(nn.Module):
    """Linear layer initialized from msmodelslim W8A8 quantized weights.

    Stores int8 weight and per-channel weight_scale. Forward dequantizes
    weight inline so that torch.onnx.export produces:
        Cast(int8→fp16) → Mul(scale) → MatMul
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer(
            "weight_scale", torch.ones(out_features, 1, dtype=dtype)
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None

        self.register_buffer(
            "input_scale", torch.ones(1, dtype=dtype)
        )
        self.register_buffer(
            "input_offset", torch.zeros(1, dtype=dtype)
        )
        self.register_buffer(
            "deq_scale", torch.zeros(out_features, dtype=torch.int64)
        )
        self.register_buffer(
            "quant_bias", torch.zeros(out_features, dtype=torch.int32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_dequant = self.weight_int8.to(x.dtype) * self.weight_scale
        return F.linear(x, weight_dequant, self.bias)


def load_w8a8_state_dict(
    safetensors_path: str,
    model: nn.Module,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cpu"),
    skip_layers: Optional[list] = None,
):
    """Replace nn.Linear layers with W8A8PreQuantizedLinear using msmodelslim weights.

    Args:
        safetensors_path: Path to the quantized safetensors file.
        model: The model with standard nn.Linear layers.
        dtype: Compute dtype (float16).
        device: Target device.
        skip_layers: Layer name patterns to skip (e.g. ["lm_head"]).

    Returns:
        The modified model with quantized linear layers.
    """
    if skip_layers is None:
        skip_layers = ["lm_head"]

    with safe_open(safetensors_path, framework="pt", device=str(device)) as f:
        all_keys = set(f.keys())

        linear_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module

        for layer_name, module in linear_layers.items():
            if any(skip in layer_name for skip in skip_layers):
                weight_key = f"{layer_name}.weight"
                if weight_key in all_keys:
                    module.weight.data = f.get_tensor(weight_key).to(dtype)
                bias_key = f"{layer_name}.bias"
                if bias_key in all_keys and module.bias is not None:
                    module.bias.data = f.get_tensor(bias_key).to(dtype)
                continue

            weight_key = f"{layer_name}.weight"
            if weight_key not in all_keys:
                continue
            weight_tensor = f.get_tensor(weight_key)
            if weight_tensor.dtype != torch.int8:
                continue

            has_bias = f"{layer_name}.bias" in all_keys
            quant_linear = W8A8PreQuantizedLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=has_bias,
                dtype=dtype,
            )

            quant_linear.weight_int8.copy_(weight_tensor)

            scale_key = f"{layer_name}.weight_scale"
            if scale_key in all_keys:
                quant_linear.weight_scale.copy_(f.get_tensor(scale_key).to(dtype))

            if has_bias:
                quant_linear.bias.copy_(f.get_tensor(f"{layer_name}.bias").to(dtype))

            input_scale_key = f"{layer_name}.input_scale"
            if input_scale_key in all_keys:
                quant_linear.input_scale.copy_(f.get_tensor(input_scale_key).to(dtype))

            input_offset_key = f"{layer_name}.input_offset"
            if input_offset_key in all_keys:
                quant_linear.input_offset.copy_(f.get_tensor(input_offset_key).to(dtype))

            deq_scale_key = f"{layer_name}.deq_scale"
            if deq_scale_key in all_keys:
                quant_linear.deq_scale.copy_(f.get_tensor(deq_scale_key))

            quant_bias_key = f"{layer_name}.quant_bias"
            if quant_bias_key in all_keys:
                quant_linear.quant_bias.copy_(f.get_tensor(quant_bias_key))

            parts = layer_name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], quant_linear.to(device))

        for key in all_keys:
            if any(
                key.endswith(suffix)
                for suffix in [
                    "weight_scale", "weight_offset", "input_scale",
                    "input_offset", "deq_scale", "quant_bias",
                ]
            ):
                continue
            if any(f"{ln}." in key for ln in linear_layers):
                continue

            parts = key.split(".")
            try:
                target = model
                for p in parts[:-1]:
                    if p.isdigit():
                        target = target[int(p)]
                    else:
                        target = getattr(target, p)
                param_name = parts[-1]
                tensor = f.get_tensor(key).to(dtype)
                if hasattr(target, param_name):
                    attr = getattr(target, param_name)
                    if isinstance(attr, (nn.Parameter, torch.Tensor)):
                        attr.data.copy_(tensor.to(attr.device))
            except (AttributeError, IndexError):
                pass

    return model
