# Qwen-Ascend-LLM

DeepSeek-R1-Distill-Qwen-1.5B 在Atlas 200I A2(310B1) 上的推理部署与性能优化。

采用 **PyTorch → ONNX → (AMCT 量化) → OM** 流程，支持 FP16 和 W8A8 量化推理。

## 目标平台

| 平台 | 芯片 | soc_version | 说明 |
|------|------|-------------|------|
| Atlas 200I A2 | Ascend 310B | `Ascend310B1` | 边缘推理 |
| Atlas 800T A3 | Ascend 910C | `Ascend910_9382` | 数据中心 |

310B1 能跑的 910 都能跑，反之不一定。除 ATC 编译 `--soc_version` 不同外，其余流程通用。

## 快速开始

### 环境准备

```bash
# 1. 创建 conda 环境并安装依赖
conda create -n qwen_ascend_cann900 python=3.10 -y
conda activate qwen_ascend_cann900
pip install -r requirements.txt

# 2. 设置 CANN 环境（需预装 CANN 9.0.0）
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 模型下载

```bash
# HuggingFace
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

# Step 2: ONNX 图优化（RoPE 融合等）
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

# Step 4: cli
cd .. && python cli_chat.py \
  --hf_model_dir $HF_MODEL_DIR \
  --om_model_path output/model.om \
  --max_prefill_length 1

# Step 5 (可选): 启动 OpenAI 兼容 API 服务
python3 server.py --config configs/deepseek_r1_1.5b_910_w8a8.json
# 测试: curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
#   -d '{"model":"DeepSeek-R1-Distill-Qwen-1.5B","messages":[{"role":"user","content":"你好"}]}'
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
python cli_chat.py --hf_model_dir $HF_MODEL_DIR --om_model_path model.om --dtype float16

# OpenAI 兼容 API 服务（修改 configs/ 下的 JSON 配置后启动）
python server.py --config configs/deepseek_r1_1.5b_910_w8a8.json
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

IFMR 算法参数使用 AMCT 默认值（也可以在 cfg 中显式指定）。

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

## 历史优化版本

### 版本列表

| 版本 | 改动范围 | modeling 文件 | change_node | KV Layout | 核心改动 | 推理验证 |
|------|---------|--------------|-------------|-----------|---------|---------|
| v0 | — | (原始 HF) | change_node.py (Trilu 修复) | BSHD | baseline | 无 modeling 文件 |
| v1 | change_node | (同 v0) | v1_rope (RoPE 融合) | BSHD | NPURotaryPositionEmbedding 替代 6 算子 RoPE | 无 modeling 文件 |
| v2 | modeling | v2_kvcache | v1_rope | BSHD | KV cache 从动态 list 改为静态预分配张量 | ✅ |
| v3 | modeling | v3_kvcache_noslice | v1_rope | BSHD | 6D cache layout，索引读取替代 StridedSlice | ✅ |
| v4 | modeling | v4_noexpand | v1_rope | BSHD | GQA: unsqueeze+broadcast 替代 repeat_kv 的 Expand | ✅ |
| **v5** | **modeling** | **v5_gate_up_fuse** | **v1_rope** | **BSHD** | **MLP: gate+up 运行时 cat 为一次 MatMul + Split** | **✅ FP16 最优** |
| v6 | modeling | v6_transpose_elim | v1_rope | **BHSD** | BHSD layout，消除 attention 前的 Transpose | ✅ (memcpy 大，不推荐) |
| v7 | modeling | v7_perlayer_kv | v1_rope | BHSD | 28 层独立 KV 输入 (31 输入，onnx2om 不支持) | 未编译 |
| v7b | modeling | v7b_split_kv | v1_rope | **BHSD** | 图内 chunk+Split 替代多输入，恢复单 KV 输入 | 未编译 |
| v8 | modeling | v8_qkv_fuse | v1_rope | BSHD | QKV 投影合并为 qkv_proj + fuse_qkv_weights() | 未编译 |
| v9 | modeling | v9_kv_slice | v1_rope | BSHD | KV 读取改用 narrow/Slice 替代索引 | ✅ (性能回退) |
| **v10** | **modeling** | **v10_gate_up_prefuse** | **v1_rope** | **BSHD** | **gate_up 权重预拼接 (fuse_gate_up_weights)，AMCT 兼容** | **✅** |
| v11 | modeling + change_node | v11_kv_inplace | v11_kv_inplace (QKV 合并) | BSHD | Where 替代 cat 做 KV 原地更新 + ONNX 级 QKV 合并 | ❌ |
| v12 | change_node | (v10 的 ONNX) | v11_kv_inplace (--skip_rope) | BSHD | v10 W8A8 + v11 change_node QKV 合并 | 待验证 |

> **KV Layout 说明**: 大部分版本使用 BSHD `[batch, seq, heads, dim]`。v6/v7/v7b 使用 BHSD `[batch, heads, seq, dim]`，
> 导出和编译时须指定 `--kv_cache_layout BHSD`，推理引擎也需对应配置。实测 BHSD kernel time 略优但 memcpy 开销更大，
> wall-clock 下 BSHD 反而更快。
>
> **change_node 说明**: v1_rope 到 v10 的 change_node 文件**完全相同**，均做 RoPE 融合 + Trilu 修复。
> v4_noexpand_310b 是 310B1 专用版，只做 Trilu 修复（310B1 不支持 RoPE 融合算子，实际上这个融合算子对于性能提升不大）。
> v11 的 change_node 新增 QKV 合并 pass（合并 q/k/v_proj 为单次 MatMul + Split）。
>
> **v5 与 v10 的关系**: v10 是 v5 的升级版，v6~v9 的实验（BHSD、per-layer KV、QKV 合并、narrow/Slice）全部回退。
> v10 相比 v5 唯一的实质升级是 AMCT 兼容性：v5 的运行时 concat 让 AMCT 把 gate_up 权重判定为动态 tensor，
> 无法预量化为 INT8；v10 的静态 parameter 让 AMCT 能正确处理，这是做 W8A8 量化的前提。
> FP16 性能两者基本一致（v5: 151 tok/s, v10: 149 tok/s）。
> v11 的 change_node 新增 QKV 合并 pass（合并 q/k/v_proj 为单次 MatMul + Split）。

### 性能对比 (910, batch=1, kv_cache=4096, decode, Device 0)

**FP16 各版本 Profiling**

| 版本 | Kernel Time | TPOT (端到端) | 吞吐 | 核心改动 |
|------|------------|--------------|------|---------|
| v2 | 7.85 ms | 8.51 ms | 117 tok/s | KV cache 静态分配 |
| v3 | 6.98 ms | 7.51 ms | 133 tok/s | 去 StridedSlice |
| v4 | 6.33 ms | 7.38 ms | 136 tok/s | GQA broadcast |
| **v5** | **6.11 ms** | **6.61 ms** | **151 tok/s** | **gate_up concat 融合 (FP16 最优)** |
| v6 | 5.97 ms | 10.05 ms | 100 tok/s | BHSD layout (memcpy 开销大) |
| v8 | 5.42 ms | 6.06 ms | 165 tok/s | QKV 合并 (存疑) |
| v9 | 8.36 ms | 9.03 ms | 111 tok/s | KV slice (回退) |
| v10 | 6.10 ms | 6.73 ms | 149 tok/s | gate_up 预拼接 (当前最优)|
| v11 | 5.47 ms | 6.82 ms | 147 tok/s | KV in-place (可能需重写推理引擎，存疑) |

**量化 + QKV 合并 (wall-clock)**

| 版本 | TPOT | 吞吐 |
|------|------|------|
| v5 FP16 | 11.40 ms | 88 tok/s |
| v5+QKV FP16 | 10.71 ms | 93 tok/s |
| v10 W8A8 | 10.28 ms | 97 tok/s |
| **v12 W8A8+QKV** | **9.87 ms** | **101 tok/s** |

### MATH500  (pass@k, k=1, 500 题, top_p=0.95, temperature=0.6)

| model | acc |
|------|--------|
| v5 FP16 | 75.8% (379/500) |
| v10 W8A8 | 70.8% (354/500) |

**FP16 by subject**

| subject | acc |
|------|--------|
| Algebra | 90.3% (112/124) |
| Number Theory | 87.1% (54/62) |
| Prealgebra | 79.3% (65/82) |
| Intermediate Algebra | 70.1% (68/97) |
| Counting & Probability | 65.8% (25/38) |
| Precalculus | 60.7% (34/56) |
| Geometry | 51.2% (21/41) |

**W8A8 by subject**

| subject | acc |
|------|--------|
| Algebra | 87.9% (109/124) |
| Number Theory | 82.3% (51/62) |
| Prealgebra | 73.2% (60/82) |
| Counting & Probability | 68.4% (26/38) |
| Intermediate Algebra | 62.9% (61/97) |
| Precalculus | 51.8% (29/56) |
| Geometry | 43.9% (18/41) |


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


### 量化场景 ATC 编译

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

本项目基于 [Tlntin/qwen-ascend-llm](https://github.com/Tlntin/qwen-ascend-llm) 开发，在此基础上进行了模型适配、量化优化。
