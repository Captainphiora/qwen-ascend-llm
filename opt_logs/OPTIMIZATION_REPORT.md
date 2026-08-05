# OM 模型推理优化报告

## 概述

本报告记录了 DeepSeek-R1-Distill-Qwen-1.5B 模型在昇腾 910 NPU 上的 4 次逐层递进优化过程。每次优化都在前一版基础上叠加新技术，最终将 Decode 速度从 112.1 tok/s 提升至 140.7 tok/s（+25.5%）。

**优化关系：** v0(baseline) → v1(+RoPE融合) → v2(+KV Cache 5D) → v3(+KV Cache 6D) → v4(+GQA broadcast)

## 一、端到端性能演进

> **注：** 算子总耗时 = profiling 采集期间所有算子执行的累计耗时。采集参数：prompt="你好，请介绍一下你自己"（约 9 tokens），max_new_tokens=20，max_prefill_length=1。因为 prefill 逐 token 处理，共执行约 9(prefill) + 20(decode) = 29 次 model forward。单次 decode forward ≈ 算子总耗时 / 29。
>
> **Benchmark 参数：** prompt="请详细介绍一下机器学习的基本概念和常用算法"（16 tokens），max_new_tokens=30，greedy sampling，3轮取平均。
> - TTFT = 从推理开始到第一个输出 token 产生的耗时（包含 16 次 prefill forward）
> - TPOT = decode 阶段每生成一个 token 的平均耗时（= decode 总时间 / (生成token数-1)）

| 版本 | 算子总耗时 | TPOT | Decode 速度 | TTFT | vs baseline | 递进提升 |
|------|-----------|------|-------------|------|-------------|---------|
| v0_baseline | 241.13ms | 8.92ms | 112.1 tok/s | 150.0ms | — | — |
| v1_rope | 233.85ms | 8.85ms | 112.9 tok/s | 150.5ms | +0.7% | +0.7% |
| v2_kvcache | 225.46ms | 8.57ms | 116.6 tok/s | 146.7ms | +4.0% | +3.3% |
| v3_kvcache_noslice | 202.92ms | 7.62ms | 131.3 tok/s | 129.8ms | +17.1% | +12.6% |
| v4_noexpand | 182.48ms | 7.11ms | 140.7 tok/s | 122.4ms | +25.5% | +7.2% |

## 二、ONNX 节点变化

### 总节点数

| 版本 | 总节点数 | 算子类型数 | 相比上版 | 相比baseline |
|------|---------|-----------|---------|-------------|
| v0_baseline | 8673 | 25 | — | — |
| v1_rope | 8337 | 25 | -336 | -336 |
| v2_kvcache | 8336 | 25 | -1 | -337 |
| v3_kvcache_noslice | 7888 | 25 | -448 | -785 |
| v4_noexpand | 6096 | 23 | -1792 | -2577 |

### 关键算子节点数变化

| ONNX Op | v0 | v1 | v2 | v3 | v4 | 变化说明 |
|---------|----|----|----|----|----|----|
| Slice | 224 | 112 | 112 | 56 | 56 | v1消除RoPE的112个Slice; v3消除KV Cache的56个Slice |
| Neg | 56 | 0 | 0 | 0 | 0 | v1 RoPE融合消除 |
| Concat | 341 | 285 | 285 | 285 | 229 | v1消除RoPE的56个; v4消除Expand相关的56个 |
| Mul | 395 | 283 | 283 | 283 | 171 | v1消除RoPE的112个; v4消除repeat_kv的112个 |
| Expand | 56 | 56 | 56 | 56 | 0 | v4 GQA broadcast完全消除 |
| Gather | 424 | 424 | 452 | 508 | 284 | v2/v3增加KV索引Gather; v4因去除Shape相关减少 |
| NPURotaryPositionEmbedding | 0 | 56 | 56 | 56 | 56 | v1新增融合算子 |
| Reshape | 225 | 225 | 226 | 226 | 170 | v4减少(不需要repeat_kv的reshape) |
| Shape | 424 | 424 | 424 | 424 | 144 | v4大幅减少(消除动态shape推导) |
| Unsqueeze | 1942 | 1942 | 1942 | 1718 | 1494 | v3/v4逐步减少 |
| Constant | 2799 | 2799 | 2769 | 2545 | 1929 | 逐步减少（辅助常量） |

## 三、算子执行耗时变化（Profiling）

### 完整算子耗时表

| 算子 | v0 耗时 | v0 占比 | v1 耗时 | v2 耗时 | v3 耗时 | v4 耗时 | v4 占比 |
|------|---------|--------|---------|---------|---------|---------|--------|
| BatchMatMulV2 | 138.77ms | 57.5% | 142.67ms | 138.60ms | 145.46ms | 138.29ms | 75.8% |
| StridedSliceD | 48.94ms | 20.3% | 41.81ms | 37.61ms | **消除** | **消除** | 0% |
| ConcatD | 20.42ms | 8.5% | 13.83ms | 7.25ms | 7.33ms | 7.39ms | 4.1% |
| Expand | 11.54ms | 4.8% | 11.01ms | 11.63ms | 11.61ms | **消除** | 0% |
| Add | 5.63ms | 2.3% | 5.85ms | 5.75ms | 5.73ms | 5.69ms | 3.1% |
| AutomaticBufferFusionOp | 4.26ms | 1.8% | 2.09ms | 2.03ms | 2.30ms | 1.95ms | 1.1% |
| Transpose | 3.86ms | 1.6% | 3.87ms | 3.95ms | 3.89ms | 3.86ms | 2.1% |
| SoftmaxV2 | 2.82ms | 1.2% | 2.54ms | 2.52ms | 2.69ms | 2.09ms | 1.1% |
| GatherV2 | 2.51ms | 1.0% | 2.46ms | 8.54ms | 16.32ms | 15.68ms | 8.6% |
| Neg | 1.78ms | 0.7% | **消除** | **消除** | **消除** | **消除** | 0% |
| RotaryPositionEmbedding | — | — | 6.19ms | 6.02ms | 5.96ms | 5.90ms | 3.2% |
| RealDiv | — | — | 0.98ms | 1.00ms | 1.00ms | 0.99ms | 0.5% |
| **总计** | **241.13ms** | | **233.85ms** | **225.46ms** | **202.92ms** | **182.48ms** | |

### 算子调用次数变化

| 算子 | v0 次数 | v1 次数 | v2 次数 | v3 次数 | v4 次数 | 变化原因 |
|------|--------|--------|--------|--------|--------|---------|
| BatchMatMulV2 | 7337 | 7337 | 7337 | 7337 | 7337 | 不变（核心计算） |
| StridedSliceD | 4872 | 1624 | 1624 | 0 | 0 | v1消除RoPE的3248次; v3消除KV的1624次 |
| ConcatD | 3277 | 1653 | 1653 | 1653 | 1653 | v1消除RoPE的1624次; 剩余为KV Cache拼接 |
| Expand | 1624 | 1624 | 1624 | 1624 | 0 | v4完全消除 |
| GatherV2 | 1653 | 1653 | 2465 | 4089 | 4089 | v2/v3因KV索引增加 |
| Neg | 1624 | 0 | 0 | 0 | 0 | v1 RoPE融合消除 |
| RotaryPositionEmbedding | 0 | 1624 | 1624 | 1624 | 1624 | v1新增（替代7节点子图） |
| 算子调用总次数 | ~26400 | ~20500 | ~21518 | ~21518 | ~19894 | 逐步减少 |

## 四、每版优化详解

### v1_rope: RoPE 融合

**在 v0 基础上新增的改动：** 仅修改 ONNX 后处理脚本（change_node），不改 modeling。

**消除的算子模式（每层 Q/K 各一次，共 56 处）：**

> 下面用 `算子(输入) → 输出名` 表示 ONNX 图中的数据流。

```
输入: x (Q或K投影后的张量, shape [1, heads, seq, 128])

Slice(x, dim=-1, [0:64])   → x1       # 取前半维度 x[..., :64]
Slice(x, dim=-1, [64:128]) → x2       # 取后半维度 x[..., 64:]
Neg(x2)                    → neg_x2   # 对后半取负 -x2
Concat(neg_x2, x1, dim=-1) → rot_x    # 拼接为 [-x2, x1], 即 rotate_half(x)
Mul(x, cos_embed)          → a        # x 乘 cos 位置编码
Mul(rot_x, sin_embed)      → b        # rotate_half(x) 乘 sin 位置编码
Add(a, b)                  → output   # 最终结果: x*cos + rotate_half(x)*sin
```
**替换为：** `NPURotaryPositionEmbedding(x, cos_embed, sin_embed) → output` 单算子，7→1

**净效果：**
- 消除: 112 Slice + 56 Neg + 56 Concat + 112 Mul + 56 Add = 392 节点
- 新增: 56 NPURotaryPositionEmbedding
- 净减少: 336 节点
- 耗时: StridedSliceD 从 48.94ms→41.81ms (少了RoPE的3248次调用)

### v2_kvcache: KV Cache 5D 重构

**在 v1 基础上新增的改动：** 修改 modeling 文件中的 KV Cache 访问方式。

**原始方式（v0/v1）：**
```python
past_key_values.transpose(1, 2)  # [1, 112, kv_len, 128]
# 每层: past_key_value[:, layer_idx*4 : (layer_idx+1)*4]  ← 大范围StridedSlice
```

**v2 方式：**
```python
past_key_values.permute(0,2,1,3).view(1, 28, 4, kv_len, 128)  # 5D
# 每层: past_key_value[:, layer_idx]                           ← Gather(常量)
#       layer_kv[:, :2] / layer_kv[:, 2:]                     ← 小范围Slice(仍存在)
```

**净效果：**
- StridedSliceD 单次耗时变化: 10us→23us（shape变了），但总次数不变(1624)
- ConcatD: 13.83ms→7.25ms（-47.6%，输入张量从112-head变为4-head）
- GatherV2: 2.46ms→8.54ms（+6ms，新增per-layer索引开销）
- 算子总耗时: 233.85→225.46ms（-3.6%）

### v3_kvcache_noslice: KV Cache 6D 完全消除 Slice

**在 v2 基础上新增的改动：** 将 5D reshape 改为 6D，消除 K/V 拆分的 Slice。

**v2 方式（仍有 Slice）：**
```python
view(1, 28, 4, kv_len, 128)     # 5D
layer_kv[:, :2]  / [:, 2:]      # Slice拆分K/V ← StridedSliceD 1624次
```

**v3 方式（纯 Gather）：**
```python
view(1, 28, 2, 2, kv_len, 128)  # 6D
past_key_value[:, layer_idx, 0]  # Gather(常量) 获取K
past_key_value[:, layer_idx, 1]  # Gather(常量) 获取V
```

**净效果：**
- StridedSliceD: 37.61ms → **完全消除**（节省 37.61ms）
- GatherV2: 8.54ms→16.32ms（+7.78ms，增加了K/V两次Gather）
- 净收益: 37.61 - 7.78 = 29.83ms
- 算子总耗时: 225.46→202.92ms（-10.0%）

### v4_noexpand: GQA Broadcast 消除 Expand

**在 v3 基础上新增的改动：** 用 grouped broadcast MatMul 替代 repeat_kv 的 Expand。

**v3 方式（有 Expand）：**
```python
key_states = repeat_kv(key_states, 6)   # [1,2,kv,128] expand→ [1,12,kv,128]
attn = matmul(Q, K^T)                   # [1,12,q,kv]
```

**v4 方式（broadcast）：**
```python
Q = Q.view(1, 2, 6, q, 128)            # 分组视图
K = K.unsqueeze(2)                      # [1, 2, 1, kv, 128]
attn = matmul(Q, K.transpose(3,4))      # broadcast: [1,2,6,q,kv]
```

**净效果：**
- Expand: 11.61ms → **完全消除**
- ONNX 节点: 7888→6096（-1792），大量Shape/Unsqueeze/Constant随之消除
- 算子总耗时: 202.92→182.48ms（-10.1%）
- BatchMatMulV2 耗时从 145.46→138.29ms（因为分组broadcast的MatMul shape更小，效率更高）

## 五、优化 tradeoff 分析

| 优化 | 收益 | 代价 | 净收益 |
|------|------|------|--------|
| v1 RoPE 融合 | 消除Neg/Slice/Concat (约13ms) | 新增RotaryPositionEmbedding (6.19ms) | ~7ms |
| v2 KV Cache 5D | ConcatD -6.6ms, StridedSliceD -4.2ms | GatherV2 +6.1ms | ~4.7ms |
| v3 KV Cache 6D | StridedSliceD -37.6ms | GatherV2 +7.8ms | ~29.8ms |
| v4 GQA broadcast | Expand -11.6ms, 节点大幅减少 | 无明显代价 | ~20.4ms |

**总净收益：** 241.13 - 182.48 = **58.65ms**（-24.3%）

## 六、算子分布饼图演进

### v0_baseline 算子耗时分布
```
BatchMatMulV2    ████████████████████████████░░░░░░░░░░░░░░░░░  57.5%
StridedSliceD    ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  20.3%
ConcatD          ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   8.5%
Expand           ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   4.8%
其他             ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   8.9%
```

### v4_noexpand 算子耗时分布
```
BatchMatMulV2    ██████████████████████████████████████░░░░░░░  75.8%
GatherV2         ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   8.6%
ConcatD          ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   4.1%
RoPE             █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3.2%
其他             ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   8.3%
```

优化后 BatchMatMulV2 的占比从 57.5% 提升到 75.8%——说明非计算类算子的开销已基本被消除，模型推理已接近 compute-bound 状态。

## 七、结论

通过 4 次逐层递进优化，系统性地消除了 ONNX 图中的 3 类主要开销算子：

1. **StridedSliceD**（原占 20.3%）→ 完全消除
2. **Expand**（原占 4.8%）→ 完全消除
3. **ConcatD**（原占 8.5%）→ 降至 4.1%（剩余为不可消除的 KV Cache seq 拼接）

最终模型在推理时的算力利用率显著提升——75.8% 的时间花在核心 MatMul 计算上，非计算开销仅占 24.2%（其中 8.6% 为 Gather 索引，是 KV Cache 重构引入的必要代价）。
