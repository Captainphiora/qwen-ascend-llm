# AGENTS.md

## 项目概述

PyTorch → ONNX → OM 流程部署 DeepSeek-R1-Distill-Qwen-1.5B 大模型推理。
目标平台: Atlas 200I A2 (310B1) / Atlas 800T A3 (910)。

## 目录结构

```
├── config.py                  # 推理配置
├── main.py / api.py / server.py / cli_chat.py  # 入口和服务
├── utils/                     # 推理引擎 (engine/session/kvcache/inference)
├── export/                    # PyTorch → ONNX → OM 导出代码
│   ├── modeling_qwen2_v*.py   # 各版本模型定义 (v2~v11)
│   ├── change_node_v*.py      # ONNX 图改写（RoPE 融合、QKV 合并等）
│   ├── export_onnx.py         # ONNX 导出
│   └── onnx2om.py             # ATC 编译封装
├── scripts/                   # 工具脚本（量化、验证、profiling）
├── benchmarks/                # 性能测试
├── docs/                      # 文档
├── client/                    # OpenAI 客户端
├── configs/                   # 模型配置
└── tests/                     # 测试
```

## 编码规范

- Python 代码须兼容 AArch64 + CANN 9.0 环境
- 所有路径使用相对路径或 `os.path.join()`，不硬编码绝对路径
- 性能数据必须标注测试条件（模型版本、KV长度、输入长度、soc_version）
- 不提交 `kernel_meta/`、`__pycache__/`、`*.om`、`*.onnx`、profiling 原始数据

## 关键注意事项

### ONNX 导出

- 必须用 NPU + FP16: `--device_str npu --dtype float16`
- CPU 导出只能 FP32，ATC 编译时会插入大量 Cast 算子

### 量化

- ATC 编译量化模型必须 `--precision_mode origin`
- 校准脚本的 attention_mask 构造必须与 `utils/kvcache.py` 一致
- gate_up 权重需预拼接（v10 modeling），否则 AMCT 无法正确预量化

### ATC 编译

- soc_version: 910 用 `Ascend910_9382`，310B1 用 `Ascend310B1`
- 310B1 不支持 RoPE 融合算子，需用 `change_node_v4_noexpand_310b.py`
