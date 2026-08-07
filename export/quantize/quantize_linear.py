"""
PyTorch 侧 W8A8 量化模块 (方案C)
将 nn.Linear 替换为量化版本, 导出ONNX时会生成 Cast→INT8 节点,
经 change_node 脚本转换为 AscendQuant 算子, atc 可直接编译。

工作原理:
  原始: Y = X @ W^T + bias
  量化后: Y = (cast_to_int8(X / x_scale) * x_scale) @ (cast_to_int8(W / w_scale) * w_scale)^T + bias
  
  导出ONNX后, Cast(to=INT8) 节点会被 change_node 替换为 AscendQuant(scale, offset)

用法:
  from export.quantize.quantize_linear import quantize_model
  quantize_model(model, cfg=quantize_cfg)
  # 然后正常 torch.onnx.export(model, ...)

quantize_cfg 格式:
  {
      "q_proj": {"type": "W8X8", "act_scale": False},
      "k_proj": {"type": "W8X8", "act_scale": False},
      "v_proj": {"type": "W8X8", "act_scale": False},
      "o_proj": {"type": "W8X8", "act_scale": False},
      "gate_proj": {"type": "W8X8", "act_scale": False},
      "up_proj": {"type": "W8X8", "act_scale": False},
      "down_proj": {"type": "W8X8", "act_scale": False},
  }
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class W8X8Linear(nn.Module):
    """
    W8A8 量化 Linear 层
    权重静态量化为INT8 (per-channel symmetric)
    激活动态量化为INT8 (per-tensor symmetric)
    
    ONNX导出时, 量化/反量化操作会表达为:
      Cast(x, to=INT8) 和 Cast(x, to=FLOAT16)
    change_node脚本会将 Cast→INT8 转换为 AscendQuant 算子
    """

    def __init__(self, original_linear: nn.Linear, act_scale: bool = False):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.has_bias = original_linear.bias is not None
        self.act_scale = act_scale

        weight = original_linear.weight.data.float()
        w_abs_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        self.w_scale = nn.Parameter(
            (w_abs_max / 127.0).to(original_linear.weight.dtype),
            requires_grad=False
        )
        weight_int8 = torch.clamp(
            torch.round(weight / w_abs_max * 127.0),
            -128, 127
        ).to(torch.int8)
        self.register_buffer("weight_int8", weight_int8)

        if self.has_bias:
            self.bias = nn.Parameter(original_linear.bias.data, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_dequant = self.weight_int8.to(x.dtype) * self.w_scale

        if self.act_scale:
            x_abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            x_scale = x_abs_max / 127.0
            x_quant = torch.clamp(torch.round(x / x_scale), -128, 127).to(torch.int8)
            x_dequant = x_quant.to(x.dtype) * x_scale
        else:
            x_dequant = x

        output = torch.nn.functional.linear(x_dequant, weight_dequant, self.bias)
        return output


class W8A16Linear(nn.Module):
    """
    W8A16 量化 Linear 层 (仅权重量化, 更安全)
    权重静态量化为INT8 (per-channel symmetric)
    激活保持原始精度 (FP16)
    """

    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.has_bias = original_linear.bias is not None

        weight = original_linear.weight.data.float()
        w_abs_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        self.w_scale = nn.Parameter(
            (w_abs_max / 127.0).to(original_linear.weight.dtype),
            requires_grad=False
        )
        weight_int8 = torch.clamp(
            torch.round(weight / w_abs_max * 127.0),
            -128, 127
        ).to(torch.int8)
        self.register_buffer("weight_int8", weight_int8)

        if self.has_bias:
            self.bias = nn.Parameter(original_linear.bias.data, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_dequant = self.weight_int8.to(x.dtype) * self.w_scale
        return torch.nn.functional.linear(x, weight_dequant, self.bias)


def quantize_model(model: nn.Module, cfg: Dict[str, Dict]) -> nn.Module:
    """
    根据配置对模型中的Linear层进行量化替换
    
    Args:
        model: 原始PyTorch模型
        cfg: 量化配置字典, key为层名后缀, value为量化参数
            {
                "q_proj": {"type": "W8X8", "act_scale": False},
                "gate_proj": {"type": "W8A16"},
                ...
            }
    Returns:
        量化后的模型 (in-place修改)
    """
    if not cfg:
        return model

    replaced_count = 0
    skipped_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        layer_suffix = name.split(".")[-1]
        matched_cfg = None
        for pattern, pattern_cfg in cfg.items():
            if layer_suffix == pattern or pattern in name:
                matched_cfg = pattern_cfg
                break

        if matched_cfg is None:
            skipped_count += 1
            continue

        quant_type = matched_cfg.get("type", "W8X8")
        act_scale = matched_cfg.get("act_scale", False)

        if quant_type == "W8X8":
            quantized_layer = W8X8Linear(module, act_scale=act_scale)
        elif quant_type == "W8A16":
            quantized_layer = W8A16Linear(module)
        else:
            print(f"  [WARN] Unknown quant type '{quant_type}' for {name}, skipping")
            skipped_count += 1
            continue

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], quantized_layer)
        replaced_count += 1

    print(f"[Quantize] Replaced {replaced_count} layers, skipped {skipped_count}")
    return model


def get_default_quantize_cfg(model_type: str = "qwen2") -> Dict[str, Dict]:
    """
    获取默认量化配置
    Args:
        model_type: 模型类型, 目前支持 "qwen2"
    """
    if model_type == "qwen2":
        return {
            "q_proj": {"type": "W8X8", "act_scale": False},
            "k_proj": {"type": "W8X8", "act_scale": False},
            "v_proj": {"type": "W8X8", "act_scale": False},
            "o_proj": {"type": "W8X8", "act_scale": False},
            "gate_proj": {"type": "W8X8", "act_scale": False},
            "up_proj": {"type": "W8X8", "act_scale": False},
            "down_proj": {"type": "W8X8", "act_scale": False},
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
