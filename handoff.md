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

## 采样（Sampling）性能优化

### 问题背景

原始 `sample_logits` 在 `utils/inference.py` 中使用 numpy 实现 top_p 采样，存在严重性能问题：
top_p 每 token 耗时 **15.7ms**，而 greedy 仅 **0.98ms**（16x 差距）。对于 310B 这种弱算力设备，采样开销可能接近甚至超过模型推理本身。

### 根因分析

| 瓶颈 | 原因 | 影响 |
|------|------|------|
| 全量 softmax（152K vocab） | 对整个词表做 exp + sum，O(n) 但常数大 | ~1.5ms |
| argpartition(k=1000) | 候选集过大，实际 p=0.8 只需 ~10 个 token | ~2ms |
| fp16 argmax（greedy） | numpy 对 float16 无 SIMD 优化路径 | 0.98ms vs 0.31ms |

### 优化方案（参考 vllm-ascend / Transformers / MindIE）

#### 主流框架对比

| 框架 | 采样位置 | 实现方式 |
|------|---------|---------|
| HuggingFace Transformers | GPU | `torch.sort` + `cumsum` + `masked_fill` + `multinomial` |
| vLLM (GPU) | GPU | FlashInfer CUDA kernel 或 PyTorch fallback |
| vllm-ascend (NPU) | NPU | triton-ascend kernel，全程不离开 NPU |
| MindIE (NPU) | NPU | CANN ATB `TopkToppSamplingOperation` 融合算子 |
| **本项目** | **CPU** | numpy argpartition + 局部 softmax（因 AclSession 已将 logits 拷回 CPU） |

#### 为什么本项目用 numpy 而非 torch

因为 `AclSession.run()` 将 logits 从 NPU 拷回 CPU 为 numpy 数组，采样发生在 Host 侧。在 CPU 上：
- `torch.sort`（O(n log n)）对 152K 词表需 **69ms**——极慢
- `np.argpartition`（O(n) 平均）仅需 **1.35ms**——快 50 倍
- 主流框架的 torch 实现仅在 GPU/NPU 上有并行优势

#### 最终实现（已提交）

```
greedy:  fp16→fp32 cast + np.argmax        → 0.31ms/token
top_k:   argpartition(k) + 局部softmax     → 1.7ms/token  
top_p:   argpartition(100) + sort(100) + cumsum截断 → 1.8ms/token
```

核心思路（与 vllm-ascend/MindIE ATB 算子一致）：
1. **先 top-k 筛选候选集**（argpartition, k=100），将 152K 词表缩减到 100 候选
2. **在候选集内做 top-p 截断**（局部 sort + cumsum），仅对 100 个元素操作
3. **Fallback 机制**：若 top-100 累积概率不足 p，扩大到 k=1000 重试

### 性能对比

| 方法 | 优化前 | 优化后 | 提速 |
|------|--------|--------|------|
| Greedy | 0.98ms | **0.31ms** | 3.2x |
| Top-k (k=50) | ~2ms | **1.7ms** | 已最优 |
| Top-p (p=0.8) | 15.7ms | **1.8ms** | 8.7x |

### 310B 适配说明

**当前代码无需修改即可在 310B 上运行**，原因：

1. 310B NPU 不支持 fp32，但采样发生在 ARM CPU（TAISHANV200M），CPU 支持 fp32 运算
2. OM 模型输出 logits 为 fp16，通过 D2H memcpy 拷回 Host 后仍为 `np.float16`
3. 代码中 `logits.astype(np.float32)` 在 ARM CPU 上执行，功能正确

**310B 上的性能预期：**

310B ARM CPU（1.6GHz, 4 核）相比 x86 服务器较弱，预计：
- argpartition 在 310B 上可能需要 5-15ms（x86 上 1.35ms）
- 若模型 decode 速度为 ~30-50ms/token（310B NPU），采样开销占比约 10-30%
- 如果采样成为瓶颈，进一步优化方向见下方

### 进一步优化方向（如 310B CPU 采样成为瓶颈）

已实现 NPU ATB 采样方案（见下节），310B 上可直接启用。其余备选方向：

| 方向 | 描述 | 难度 |
|------|------|------|
| 降低 k_candidate | 将 top-100 降为 top-50（对 p≤0.9 通常足够） | 低 |
| 避免 D2H logits 拷贝 | 修改 AclSession 保留 NPU buffer，仅拷回采样结果 | 高 |
| ARM NEON 优化 | 用 Cython/Numba 对 argpartition 做 SIMD 加速 | 中 |

### NPU ATB 采样集成（已实现）

使用 `torch_npu.npu_top_k_top_p_sample` 算子，将 top_p/top_k 采样在 NPU 上完成。

**接口说明：**

```python
torch_npu.npu_top_k_top_p_sample(
    logits,    # [batch, vocab_size], float16/bfloat16, 待采样词频
    top_k,     # [batch], int32, 1≤k≤1024
    top_p,     # [batch], float16, 0<p<1 (设≥1则跳过topP)
    q=None,    # [batch, vocab_size], float32, 随机权重(None则跳过sample)
) -> (logits_select_idx, logits_top_kp_select)
```

**设计策略（hybrid）：**
- **Greedy**: CPU argmax（简单 argmax 不值得 H2D 数据搬移开销）
- **Top_k/Top_p**: NPU ATB 算子（复杂采样在 NPU 上快 8.6x）
- 自动检测 `torch_npu` 可用性，不可用时回退 CPU numpy

**实测结果（910, DeepSeek-R1-Distill-Qwen-1.5B）：**

| 配置 | Decode 速度 | 说明 |
|------|------------|------|
| Greedy (CPU argmax) | 105.9 tok/s | 最优：H2D 拷贝开销 > argmax 计算 |
| Top_p (NPU ATB) | 94.7 tok/s | 比 CPU numpy 86.5 tok/s 快 9.5% |
| Top_p (CPU numpy) | 86.5 tok/s | fallback 路径 |

**310B 预期收益更大**：ARM CPU 弱（argpartition ~10ms），NPU ATB 采样 ~0.2ms 可节省 ~10ms/token。

**支持的硬件：** Atlas A2/A3 训练/推理系列产品（910, 310B4 等）。需确认 310B 是否在支持列表内（文档标注 "A200I A2 Box 异构组件" 支持）。

### 相关 Git 提交

```
7f6f400 feat: 集成NPU ATB采样算子 (torch_npu.npu_top_k_top_p_sample)
728ee30 perf: greedy采样优化 - fp16转fp32后argmax (0.98ms→0.31ms, 3.2x提速)
b5d9546 perf: 采样优化最终方案 - numpy argpartition+局部softmax (1.7ms/token, 比torch.sort快30x+)
1e980b7 perf: 优化top_p采样 - 避免全量softmax, 减少k_candidate (17ms→2.3ms/token)
```
