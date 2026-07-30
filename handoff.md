# 推理优化工作交接文档

## 项目概述

本项目实现了在昇腾 NPU 上使用 OM 离线模型对大语言模型进行推理。优化工作围绕 **DeepSeek-R1-Distill-Qwen-1.5B** 模型，通过 ONNX 图级别改写，在不修改底层推理引擎的前提下，将 Decode 速度从 112.1 tok/s 提升到 140.7 tok/s（+25.5%）。

**目标硬件：** Ascend 310B（Atlas 200I DK A2），当前在 Ascend 910 上验证优化方案
**分支：** `200I-DK-A2`
**CANN 版本：** 9.0.0（路径 `/usr/local/Ascend/cann-9.0.0`）

## 已完成的优化

### 最终成果（v4_noexpand）

| 指标 | baseline | 优化后 | 提升 |
|------|----------|--------|------|
| Decode 速度 | 112.1 tok/s | 140.7 tok/s | **+25.5%** |
| TPOT（每 token 延迟） | 8.92ms | 7.11ms | -20.3% |
| TTFT（首字延迟） | 150.0ms | 122.4ms | -18.4% |
| 算子总耗时 | 241.13ms | 182.48ms | -24.3% |
| ONNX 节点数 | 8673 | 6096 | -29.7% |

### 优化技术栈（逐层叠加）

#### 1. RoPE 融合（v1_rope）— 仅 910 可用

- **原理：** 将 RoPE 的 7 节点子图（2×Slice + Neg + Concat + 2×Mul + Add）替换为昇腾 `NPURotaryPositionEmbedding` 单算子
- **改动位置：** `export/change_node_v1_rope.py`（ONNX 后处理脚本）
- **局限：** `RotaryPositionEmbedding` 算子在 310B 上不支持，310B 需跳过此优化
- **收益：** 算子总耗时 -3.0%

#### 2. KV Cache 6D 重构（v3_kvcache_noslice）— 910 和 310B 均可用

- **原理：** 将 KV Cache 张量从 `[1, kv_len, 112, 128]` reshape 为 `[1, 28, 2, 2, kv_len, 128]`（6D），每层通过常量索引 Gather 获取自己的 K/V，完全消除 StridedSliceD
- **改动位置：** `export/modeling_qwen2_v3_kvcache_noslice.py`（PyTorch 模型导出文件）
- **关键改动：**
  - `Qwen2Model.forward()`: `permute(0,2,1,3).view(1, num_layers, 2, num_kv_heads, -1, head_dim)`
  - `Qwen2Attention.forward()`: `past_key_value[:, layer_idx, 0]` / `[:, layer_idx, 1]`
- **收益：** StridedSliceD 48.94ms → 0（完全消除）

#### 3. GQA Broadcast 消除 Expand（v4_noexpand）— 910 和 310B 均可用

- **原理：** 用 grouped matmul broadcast 替代 `repeat_kv` 的显式 expand。Q reshape 为 `[b, kv_heads, groups, q, d]`，K/V unsqueeze 为 `[b, kv_heads, 1, kv, d]`，MatMul 自动 broadcast
- **改动位置：** `export/modeling_qwen2_v4_noexpand.py`（继承 v3 全部改动）
- **收益：** Expand 11.61ms → 0（完全消除），额外减少 1792 ONNX 节点

### 平台差异

| 优化 | 910 | 310B |
|------|-----|------|
| KV Cache 6D 重构 | 可用 | 可用 |
| GQA Broadcast | 可用 | 可用 |
| RoPE 融合（NPURotaryPositionEmbedding） | 可用 | **不可用** |
| FlashAttention / IncreFlashAttention | 可用 | **不可用** |

310B 编译时使用专用 change_node：`export/change_node_v4_noexpand_310b.py`（仅做 Trilu/Cast 修复，不做 RoPE 融合）。

## 怎么做的（工作流）

### 整体方法论

```
Profiling 采集 → 定位瓶颈算子 → 分析算子来源 → 修改 modeling 文件 / change_node 脚本
→ 重新导出 ONNX → ATC 编译 OM → Profiling + Benchmark 验证
```

### 导出流程

```
PyTorch (modeling_qwen2_vX.py)
  → torch.onnx.export (export/export_onnx.py, opset=14, simplify=false)
  → ONNX raw (onnx_raw/)
  → change_node (export/change_node_vX.py) — 图改写
  → ONNX changed (onnx_changed/)
  → ATC 编译 (export/onnx2om.py, --soc_version=Ascend910_9382 or Ascend310B1)
  → OM 模型
```

### 一键脚本

```bash
# 910 上完整流程（导出 + 编译 + profiling + benchmark）
bash run_rope_optimize.sh --version v4_noexpand \
    --modeling_file modeling_qwen2_v4_noexpand.py \
    --change_node change_node_v4_noexpand.py \
    --device_id 5

# 310B 编译（复用已有 ONNX raw，用 310B 专用 change_node）
python export/change_node_v4_noexpand_310b.py \
    --input_model_path opt_models/v4_noexpand/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --output_model_path opt_models/v4_noexpand_310b/onnx_changed/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx

python export/onnx2om.py \
    --onnx_model_path opt_models/v4_noexpand_310b/onnx_changed/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --om_model_path ./output/model_310_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b \
    --soc_version Ascend310B1 --kv_cache_length 4096 --max_prefill_length 1 \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B
```

### Profiling 工具链

```bash
# 独立 profiling（采集 + 解析 + 分析）
bash run_profiling_all.sh --om_model_path <path> --device_id 5

# 底层脚本
profiling_collect.py  — ACL Python API 采集 profiling 原始数据
profiling_analyze.py  — 读取 msprof CSV 生成分析报告
```

## 关键文件清单

| 文件 | 用途 |
|------|------|
| `export/modeling_qwen2.py` | 原始 modeling（baseline） |
| `export/modeling_qwen2_v4_noexpand.py` | 最终优化版 modeling（KV Cache 6D + GQA broadcast） |
| `export/change_node_v4_noexpand.py` | 910 用 change_node（含 RoPE 融合） |
| `export/change_node_v4_noexpand_310b.py` | 310B 用 change_node（不含 RoPE 融合） |
| `export/change_node_v1_rope.py` | RoPE 融合独立脚本 |
| `export/export_onnx.py` | PyTorch → ONNX 导出 |
| `export/onnx2om.py` | ONNX → OM 编译（ATC 封装） |
| `run_rope_optimize.sh` | 一键导出+编译+profiling+benchmark |
| `run_profiling_all.sh` | 独立 profiling 采集脚本 |
| `profiling_collect.py` | ACL profiling 采集 |
| `profiling_analyze.py` | profiling 分析报告生成 |
| `benchmark.py` | 性能基准测试（TTFT/TPOT） |
| `opt_logs/OPTIMIZATION_RECORD.md` | 详细优化记录（含代码 diff、数据对比） |

## 环境配置

```bash
# Conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen_ascend_cann900

# CANN 环境（~/.bashrc_cann900 封装）
source ~/.bashrc_cann900
# 等价于: _cann_setup /usr/local/Ascend/cann-9.0.0 /usr/local/Ascend/nnal/atb/9.0.0/atb/set_env.sh
```

## 未来可能的优化方向

### 可行方向（310B 兼容）

| 方向 | 预期收益 | 难度 | 说明 |
|------|---------|------|------|
| INT8/INT4 量化 | 高（MatMul 减半） | 中 | 310B 支持 INT8/INT4 稀疏计算，需配合 AMCT 量化工具 |
| KV Cache in-place scatter | 中（消除 ConcatD 7.4ms） | 高 | 需修改 OM 输入输出协议和 engine.py，用 scatter 替代 concat |
| Transpose layout 优化 | 低（3.9ms） | 低 | 调整 modeling 中的 tensor layout 避免不必要的转置 |
| max_prefill_length 增大 | 中（减少 TTFT） | 低 | 改为 8/16，减少 prefill 阶段 model execute 次数 |

### 不可行方向（310B 不支持）

| 方向 | 原因 |
|------|------|
| FlashAttention / IncreFlashAttention 融合 | 310B 算子库不支持（CANN 文档明确标注 Atlas 200I/500 A2 推理产品不支持） |
| NPURotaryPositionEmbedding | 310B 不支持该融合算子（ATC 编译报 EZ3003 错误） |

### 量化方案建议（优先级最高的下一步）

310B 支持 FP16/INT8/INT4 推理。当前 v4_noexpand 中 MatMul 占 75.8%，量化可以最直接压缩这部分：

1. **W8A8 量化（权重 INT8 + 激活 INT8）：** 使用 CANN 的 AMCT（Ascend Model Compression Toolkit）对模型做训练后量化，将 Linear 层的 MatMul 替换为 INT8 MatMul
2. **W4A16 量化（权重 INT4 + 激活 FP16）：** 适合推理场景，权重存储减半，计算时动态反量化
3. 量化后需重新验证精度（可用 benchmark.py 的 greedy decode 输出对比）

### 注意事项

- 所有优化均不修改 `utils/engine.py`（推理引擎），只修改导出阶段的 modeling 文件和 ONNX 后处理脚本
- OM 模型的输入输出 shape 保持不变：`past_key_values: [1, 4096, 112, 128]`，engine 中的 memcpy 逻辑无需改动
- 310B 和 910 共享同一份 `onnx_raw`（由 `modeling_qwen2_v4_noexpand.py` 导出），区别仅在 change_node 阶段和 ATC 的 `--soc_version` 参数
