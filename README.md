# Qwen-Ascend-LLM

DeepSeek-R1-Distill-Qwen-1.5B 在昇腾 NPU 上的推理部署与性能优化。

采用 **PyTorch → ONNX → OM** 流程，支持 FP16 和 W8A8 量化推理。

## 目标平台

| 平台 | 芯片 | soc_version | 说明 |
|------|------|-------------|------|
| Atlas 200I A2 | Ascend 310B1 | `Ascend310B1` | 边缘推理 |
| Atlas 800T A3 | Ascend 910 | `Ascend910_9382` | 数据中心 |

310B1 能跑的 910 都能跑，反之不一定。除 ATC 编译时 `--soc_version` 不同外，其余流程通用。

## 快速开始

### 环境准备

```bash
# 1. 激活 conda 环境（需预先创建，包含 torch、torch_npu、onnxruntime 等依赖）
conda activate qwen_ascend_cann900

# 2. 设置 CANN 路径（根据实际安装路径修改）
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/lib64/plugin/opskernel:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PATH=$ASCEND_HOME_PATH/bin:$PATH
export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 模型下载

```bash
# HuggingFace（需网络访问）
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir models/DeepSeek-R1-Distill-Qwen-1.5B
```

### 端到端部署（FP16）

```bash
HF_MODEL_DIR=/path/to/DeepSeek-R1-Distill-Qwen-1.5B

# Step 1: PyTorch → ONNX
cd export && cp modeling_qwen2_v5_gate_up_fuse.py modeling_qwen2.py
python export_onnx.py \
  --hf_model_dir $HF_MODEL_DIR \
  --onnx_model_path output/model.onnx \
  --kv_cache_length 4096 --kv_cache_layout BSHD \
  --device_str npu --dtype float16

# Step 2: ONNX 图优化（RoPE 融合）
python change_node_v5_gate_up_fuse.py \
  --input_model_path output/model.onnx \
  --output_model_path output/model_changed.onnx

# Step 3: ATC 编译（ONNX → OM）
python onnx2om.py \
  --hf_model_dir $HF_MODEL_DIR \
  --onnx_model_path output/model_changed.onnx \
  --om_model_path output/model \
  --kv_cache_length 4096 --max_prefill_length 1 \
  --kv_cache_layout BSHD \
  --soc_version Ascend310B1

# Step 4: 推理
cd .. && python cli_chat.py \
  --hf_model_dir $HF_MODEL_DIR \
  --om_model_path output/model.om \
  --max_prefill_length 1
```

---

## W8A8 量化

## 完整部署流程

### 1. PyTorch → ONNX 导出

`export/export_onnx.py` 将 HuggingFace 模型导出为 ONNX，支持多个优化版本的 modeling 文件。

```bash
# 使用指定版本的 modeling
cp export/modeling_qwen2_<version>.py export/modeling_qwen2.py
python export/export_onnx.py \
  --hf_model_dir $HF_MODEL_DIR \
  --onnx_model_path <output.onnx> \
  --kv_cache_length 4096 --kv_cache_layout BSHD \
  --device_str npu --dtype float16
```

**注意**: ONNX 导出必须用 NPU + FP16。CPU 导出只能用 FP32，ATC 编译时会插入大量 Cast 算子。

### 2. ONNX 图优化 (change_node)

将 ONNX 中的 RoPE 算子模式替换为昇腾 NPURotaryPositionEmbedding 融合算子。

| 平台 | 脚本 | 说明 |
|------|------|------|
| 910 | `change_node_v5_gate_up_fuse.py` | RoPE → NPURotaryPositionEmbedding |
| 310B1 | `change_node_v4_noexpand_310b.py` | Trilu 修复（310B1 不支持 RoPE 融合） |

### 3. ATC 编译 (ONNX → OM)

```bash
python export/onnx2om.py \
  --hf_model_dir $HF_MODEL_DIR \
  --onnx_model_path <changed.onnx> \
  --om_model_path <output_prefix> \
  --kv_cache_length 4096 --max_prefill_length 1 \
  --kv_cache_layout BSHD \
  --soc_version Ascend310B1  # 或 Ascend910_9382
```

### 4. 推理与服务

```bash
# CLI 交互
python cli_chat.py --hf_model_dir $HF_MODEL_DIR --om_model_path model.om

# OpenAI 兼容 API 服务
python main.py --hf_model_dir $HF_MODEL_DIR --om_model_path model.om
```

---

### 量化流程

```
FP16 ONNX (gate_up 预拼接)
    │  Step 1: AMCT 校准 (amct_onnx, 50 条 boolq 数据)
    ▼
deploy ONNX (INT8) + fake_quant ONNX (FP16 仿真)
    │  Step 2: 精度验证 (fake_quant vs FP16 baseline)
    │  Step 3: change_node (RoPE 融合)
    ▼
changed ONNX
    │  Step 4: ATC 编译 (precision_mode=origin)
    ▼
OM 模型
```

### AMCT 校准

```bash
# 需要 gate_up 预拼接的 ONNX（v10 modeling）
cp export/modeling_qwen2_v10_gate_up_prefuse.py export/modeling_qwen2.py
# 导出得到 raw ONNX，然后校准:
python scripts/amct_onnx_calibrate.py \
  --model_path <raw.onnx> \
  --hf_model_dir $HF_MODEL_DIR \
  --output_dir output/amct \
  --num_samples 50 \
  --kv_cache_length 4096 \
  --cpu_threads 64 \
  --kv_cache_layout BSHD \
  --quant_cfg scripts/quant_v8_skip60_gate_up_quantized.cfg
```

产物:
- `model_deploy_deploy_model.onnx` — INT8 部署模型（给 ATC 编译）
- `model_deploy_fake_quant_model.onnx` — FP16 仿真模型（给精度验证）
- `record.txt` — 每层的 scale_d / scale_w / offset_d

### 量化配置 (.cfg)

`scripts/quant_v8_skip60_gate_up_quantized.cfg` 控制哪些层量化、哪些跳过。

模型共 225 个 MatMul，跳过 60 层，量化 165 层:

| 跳过类型 | 数量 | 原因 |
|---------|------|------|
| lm_head | 1 | 输出层精度敏感 |
| Attention QK^T | 28 | activation x activation，无固定权重 |
| Attention Score x V | 28 | activation x activation，无固定权重 |
| outlier down_proj (L2/L26/L27) | 3 | 激活存在极端离群值，有效动态范围 < 100 |

跳过判据基于**有效动态范围** (`255 / scale_d`):
- < 100: 必须跳过（量化噪声远超信号）
- 100~500: 建议跳过
- > 500: 可接受

IFMR 算法参数使用 AMCT 默认值（不需要在 cfg 中显式指定）。

### 精度验证

```bash
# 单 prompt 快速验证
python scripts/verify_fakequant.py \
  --fp16_onnx <v5_fp16_raw.onnx> \
  --fakequant_onnx <fake_quant.onnx> \
  --hf_model_dir $HF_MODEL_DIR

# 多 prompt 全 position 验证
python scripts/verify_fakequant_v2.py \
  --fp16_onnx <v5_fp16_raw.onnx> \
  --fakequant_onnx label=<fake_quant.onnx> \
  --hf_model_dir $HF_MODEL_DIR
```

### 量化模型 ATC 编译

量化模型必须使用 `--precision_mode origin`，不能用 `mixed_float16`:

```bash
python export/onnx2om.py \
  --onnx_model_path <deploy.onnx 经 change_node 后> \
  --om_model_path <output_prefix> \
  --precision_mode origin \
  --soc_version Ascend310B1
```

### QKV 投影合并（可选，进一步优化）

在 AMCT 之前将 q/k/v_proj 合并为 qkv_proj，减少 kernel launch:

```bash
python export/change_node_v11_kv_inplace.py \
  --input_model_path <raw.onnx> \
  --output_model_path <qkv_merged.onnx> \
  --skip_rope  # 只做 QKV 合并，不做 RoPE 融合（AMCT 兼容）
```

然后对 qkv_merged ONNX 跑 AMCT 校准 -> change_node(RoPE) -> ATC 编译。

---

## 优化版本演进

### 版本列表

| 版本 | modeling 文件 | 核心改动 |
|------|--------------|---------|
| v0 | (原始 HF) | baseline |
| v1 | -- | RoPE change_node 融合 |
| v2 | v2_kvcache | KV cache 静态分配 |
| v3 | v3_kvcache_noslice | KV cache 去 slice |
| v4 | v4_noexpand | GQA 去 expand（broadcast 替代 repeat_kv）|
| **v5** | **v5_gate_up_fuse** | **gate_proj + up_proj 运行时 concat 融合** |
| v6 | v6_transpose_elim | BHSD layout（消除 transpose）|
| v7 | v7_perlayer_kv | KV cache 按层预切分 |
| v7b | v7b_split_kv | 图内 Split 替代多输入 |
| v8 | v8_qkv_fuse | QKV 投影合并（change_node 层面）|
| v9 | v9_kv_slice | KV cache slice 优化 |
| **v10** | **v10_gate_up_prefuse** | **gate_up 权重预拼接（AMCT 兼容）+ W8A8** |
| v11 | v11_kv_inplace | KV in-place + QKV 合并 |
| **v12** | (v10 + v11 QKV) | **v10 W8A8 + QKV 合并（当前最优）** |

### 性能对比 (910, batch=1, kv_cache=4096, decode)

| 版本 | 量化 | QKV合并 | TPOT | 吞吐 |
|------|------|---------|------|------|
| v5 FP16 | -- | -- | 11.40 ms | 88 tok/s |
| v5+QKV FP16 | -- | Yes | 10.71 ms | 93 tok/s |
| v10 W8A8 | W8A8 | -- | 10.28 ms | 97 tok/s |
| **v12 W8A8+QKV** | **W8A8** | **Yes** | **9.87 ms** | **101 tok/s** |

### MATH500 精度 (50 题, top_p=0.95, temperature=0.6)

| 模型 | 准确率 |
|------|--------|
| v5 FP16 | 72.0% (36/50) |
| v12 W8A8+QKV | 62.0% (31/50) |

详细结果见 `docs/` 目录下的分析文档。

---

## 项目结构

```
├── config.py                  # 推理配置
├── main.py / api.py / server.py / cli_chat.py  # 入口和服务
├── utils/                     # 推理引擎
│   ├── engine.py              # ACL 推理封装
│   ├── inference.py           # 完整推理流程
│   ├── kvcache.py             # KV cache 管理
│   └── session.py             # ONNX/ACL session
├── export/                    # PyTorch → ONNX → OM
│   ├── modeling_qwen2_v*.py   # 各版本模型定义 (v2~v11)
│   ├── change_node_v*.py      # 各版本 ONNX 图改写
│   ├── export_onnx.py         # ONNX 导出
│   └── onnx2om.py             # ATC 编译封装
├── scripts/                   # 工具脚本
│   ├── amct_onnx_calibrate.py # AMCT 量化校准
│   ├── verify_fakequant*.py   # 量化精度验证
│   ├── quant_v8_skip60_*.cfg  # 量化配置
│   └── calc_theoretical_throughput.py  # 理论吞吐分析
├── benchmarks/                # 性能测试
│   ├── benchmark.py           # 推理 benchmark
│   ├── benchmark_math500*.py  # MATH500 精度测试
│   └── parse_profiling.py     # profiling 分析
├── docs/                      # 文档
├── client/                    # OpenAI 客户端
├── configs/                   # 模型配置
└── tests/                     # 测试
```

---

## 已知问题与注意事项

### CANN 版本

多个 CANN 版本并存时，**不要** `source set_env.sh`（可能指向错误版本）。手动设置 `ASCEND_HOME_PATH` 等环境变量。

### ONNX 导出

必须使用 `--device_str npu --dtype float16`。CPU 导出的 FP32 ONNX 在 ATC 编译时会插入大量 Cast 算子，严重拖慢推理。

### 量化模型 ATC 编译

必须使用 `--precision_mode origin`，不能用 `mixed_float16`。

### 量化校准输入构造

校准时的 `attention_mask` 必须与推理引擎 (`utils/kvcache.py`) 的构造方式一致:
- ONNX 的 kv_len 维度不能为 0，推理引擎用 `kv_len = real_kv_size + 1` 补一个空 slot
- `attention_mask` 的第 0 位（空 KV slot）必须为 0: `[0, 1, 1, ..., 1]`
- 可用 `scripts/verify_calibration_inputs.py` 验证校准输入与推理引擎的一致性

### gate_up 预拼接（v10）

v5 的 ONNX 中 gate_proj 和 up_proj 通过 Concat 节点拼接。AMCT 将 Concat 输出判定为动态 tensor，无法预量化为 INT8 常量。v10 在 PyTorch 层面预拼接为 `gate_up_weight` initializer，解决此问题。

---

## 致谢

本项目参考了 [ascend-llm](https://gitee.com/yinghuo302/ascend-llm) 项目。
