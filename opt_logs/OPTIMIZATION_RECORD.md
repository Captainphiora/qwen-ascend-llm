# RoPE 推理优化记录

## 模型信息

- 模型: DeepSeek-R1-Distill-Qwen-1.5B
- KV Cache 长度: 4096
- Max Prefill Length: 1
- 硬件: Ascend 910 (Device 5)
- CANN: 9.0.0
- 分支: 200I-DK-A2

## 版本对比

### 性能结果

| 版本 | 算子总耗时 | TPOT | Decode 速度 | TTFT | vs baseline |
|------|-----------|------|-------------|------|-------------|
| v0_baseline | 241.13ms | 8.92ms | 112.1 tok/s | 150.0ms | — |
| v1_rope | 233.85ms | 8.85ms | 112.9 tok/s | 150.5ms | +0.7% |
| v2_kvcache | 225.46ms | 8.57ms | 116.6 tok/s | 146.7ms | +4.0% |
| v3_kvcache_noslice | **202.92ms** | **7.62ms** | **131.3 tok/s** | **129.8ms** | **+17.1%** |
| v4_noexpand | **182.48ms** | **7.11ms** | **140.7 tok/s** | **122.4ms** | **+25.5%** |

### 关键算子对比

| 算子 | v0_baseline | v1_rope | v2_kvcache | v3_kvcache_noslice |
|------|-------------|---------|------------|--------------------|
| BatchMatMulV2 | 138.77ms (7337次) | 142.67ms (7337次) | 138.60ms (7337次) | 145.46ms (7337次) | 138.29ms (7337次) |
| StridedSliceD | 48.94ms (4872次) | 41.81ms (1624次) | 37.61ms (1624次) | **消除** | **消除** |
| ConcatD | 20.42ms (3277次) | 13.83ms (1653次) | 7.25ms (1653次) | 7.33ms (1653次) | 7.39ms (1653次) |
| Neg | 1.78ms (1624次) | 消除 | 消除 | 消除 | 消除 |
| RotaryPositionEmbedding | — | 6.19ms (1624次) | 6.02ms (1624次) | 5.96ms (1624次) | 5.90ms (1624次) |
| GatherV2 | 2.51ms (1653次) | 2.46ms (1653次) | 8.54ms (2465次) | 16.32ms (4089次) | 15.68ms (4089次) |
| Expand | 11.54ms (1624次) | 11.01ms (1624次) | 11.63ms (1624次) | 11.61ms (1624次) | **消除** |
| Transpose | 3.86ms (29次) | 3.87ms (29次) | 3.95ms (29次) | 3.89ms (29次) | 3.86ms (29次) |

## 各版本方案说明

### v0_baseline（基线）

**使用文件:**
- modeling: `export/modeling_qwen2.py`
- change_node: `export/change_node.py`

**做法:**
- 原始 PyTorch 模型导出 ONNX, KV Cache 作为统一张量 `[1, kv_len, num_layers*2*num_kv_heads, head_dim]`
- 每层通过 `past_key_value[:, layer_idx*2*H : (layer_idx*2+1)*H]` 切片获取 K/V (产生 StridedSliceD)
- RoPE 保持原始 `Slice+Neg+Concat+Mul+Add` 模式
- change_node 仅修复 Trilu (加 upper=0) 和 Cast→INT8 替换为 AscendQuant

**复现命令:**
```bash
bash run_rope_optimize.sh --version v0_baseline \
    --modeling_file modeling_qwen2.py \
    --change_node change_node.py \
    --skip_export \
    --onnx_input output/onnx_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --device_id 5
```

### v1_rope（RoPE 融合）

**使用文件:**
- modeling: `export/modeling_qwen2.py`（同 baseline）
- change_node: `export/change_node_v1_rope.py`

**做法:**
- modeling 文件不变, ONNX 导出后通过 change_node 在图级别做 RoPE 融合
- 匹配 RoPE 子图模式: `Slice(x,:half) + Slice(x,half:) + Neg + Concat + Mul(x,cos) + Mul(rot,sin) + Add`
- 替换为单个 `NPURotaryPositionEmbedding(x, cos, sin, mode=0)` 融合算子
- 每层 Q 和 K 各有一个 RoPE, 共 28×2=56 个 pattern 被替换
- 同时保留 Trilu/Cast 修复
- 节点数: 8673 → 8337 (减少 336 节点)

**效果分析:**
- RoPE 融合消除了每层 Q/K 的 `2×Slice + Neg + Concat` 共 4 个算子 (×56 = 224 节点)
- StridedSliceD 从 4872 次降至 1624 次 (消除了 RoPE 相关的 3248 次 Slice)
- ConcatD 从 3277 次降至 1653 次 (消除了 RoPE rotate_half 的 1624 次 Concat)
- Neg 算子完全消除 (原 1624 次, 1.78ms)
- 新增 RotaryPositionEmbedding 融合算子 1624 次, 耗时 6.19ms (平均 3.8us/次)
- 算子总耗时减少 7.28ms (3.0%), 但端到端 TPOT 仅提升 0.8%, 因为融合算子本身也有开销

**复现命令:**
```bash
bash run_rope_optimize.sh --version v1_rope \
    --modeling_file modeling_qwen2.py \
    --change_node change_node_v1_rope.py \
    --skip_export \
    --onnx_input output/onnx_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --device_id 5
```

### v2_kvcache（KV Cache 重构 + RoPE 融合）

**使用文件:**
- modeling: `export/modeling_qwen2_v2_kvcache.py`
- change_node: `export/change_node_v2_kvcache.py`

**做法:**

1. **KV Cache 访问重构** (modeling 层面):
   - 原来: `past_key_values.transpose(1,2)` 后每层用大范围 Slice 从 112-head 维度中切出自己的 4 个 head
   - 现在: `past_key_values.permute(0,2,1,3).view(1, num_layers, 2*num_kv_heads, kv_len, head_dim)` 重塑为 5D 张量
   - 每层通过 `past_key_value[:, self.layer_idx]` (Gather 常量索引) 直接访问, 无需大范围 StridedSlice
   - K/V 拆分改为对小维度 (4 heads) 做 `layer_kv[:, :num_kv_heads]` 和 `layer_kv[:, num_kv_heads:]`

2. **RoPE 融合** (change_node 层面):
   - 同 v1_rope, 56 个 RoPE pattern 替换为 NPURotaryPositionEmbedding

**关键代码改动 (`modeling_qwen2_v2_kvcache.py`):**

Qwen2Model.forward() 中:
```python
# 原来:
past_key_values = past_key_values.transpose(1, 2)
# 改为:
past_key_values = past_key_values.permute(0, 2, 1, 3)
past_key_values = past_key_values.view(1, num_layers, 2 * num_kv_heads, -1, head_dim)
```

Qwen2Attention.forward() 中:
```python
# 原来:
cache_key = past_key_value[:, layer_idx*2*H : (layer_idx*2+1)*H]
cache_value = past_key_value[:, (layer_idx*2+1)*H : (layer_idx*2+2)*H]
# 改为:
layer_kv = past_key_value[:, self.layer_idx]
cache_key = layer_kv[:, :self.num_key_value_heads]
cache_value = layer_kv[:, self.num_key_value_heads:]
```

**复现命令 (需要重新导出 ONNX):**
```bash
bash run_rope_optimize.sh --version v2_kvcache \
    --modeling_file modeling_qwen2_v2_kvcache.py \
    --change_node change_node_v2_kvcache.py \
    --device_id 5
```

**效果分析:**
- 张量 shape 变化: `[1, kv_len, 112, 128]` → permute+view → `[1, 28, 4, kv_len, 128]`
- 每层用 Gather(常量索引) 替代原来对 112-head 维度的大范围 StridedSlice
- StridedSliceD: 48.94ms → 37.61ms (仍有 1624 次, 来自 K/V 拆分的小维度 slice, 但单次耗时从 10us 变为 23us 因为 shape 变了)
- ConcatD: 20.42ms → 7.25ms (**-64.5%**), 因为 concat 的输入从 112-head 大张量变为 4-head 小张量
- 代价: GatherV2 从 2.51ms (1653次) 增加到 8.54ms (2465次), 增加约 6ms, 这是 per-layer 索引带来的开销
- 净收益: 算子总耗时从 241.13ms 降至 225.46ms, 减少 15.67ms (6.5%)
- 端到端: Decode 速度 112.1 → 116.6 tok/s, 提升 4.0%; TTFT 150.0 → 146.7ms, 改善 2.2%

### v3_kvcache_noslice（完全消除 StridedSliceD + RoPE 融合）

**使用文件:**
- modeling: `export/modeling_qwen2_v3_kvcache_noslice.py`
- change_node: `export/change_node_v3_kvcache_noslice.py`

**做法:**

1. **KV Cache 6D 重构** (modeling 层面):
   - 在 v2 基础上进一步将 5D `[1, num_layers, 2*num_kv_heads, kv_len, head_dim]` 拆解为 6D `[1, num_layers, 2, num_kv_heads, kv_len, head_dim]`
   - 每层 K 通过 `past_key_value[:, self.layer_idx, 0]` 获取 (Gather 常量索引)
   - 每层 V 通过 `past_key_value[:, self.layer_idx, 1]` 获取 (Gather 常量索引)
   - 完全消除 StridedSliceD — 不再需要任何维度上的 Slice 操作

2. **RoPE 融合** (change_node 层面):
   - 同 v1_rope/v2, 56 个 RoPE pattern 替换为 NPURotaryPositionEmbedding

**关键代码改动 (`modeling_qwen2_v3_kvcache_noslice.py`):**

Qwen2Model.forward() 中:
```python
# v2 的 5D:
past_key_values = past_key_values.permute(0, 2, 1, 3)
past_key_values = past_key_values.view(1, num_layers, 2 * num_kv_heads, -1, head_dim)
# v3 改为 6D:
past_key_values = past_key_values.permute(0, 2, 1, 3)
past_key_values = past_key_values.view(1, num_layers, 2, num_kv_heads, -1, head_dim)
```

Qwen2Attention.forward() 中:
```python
# v2 (仍需小维度 Slice):
layer_kv = past_key_value[:, self.layer_idx]
cache_key = layer_kv[:, :self.num_key_value_heads]
cache_value = layer_kv[:, self.num_key_value_heads:]
# v3 (纯 Gather, 无 Slice):
cache_key = past_key_value[:, self.layer_idx, 0]
cache_value = past_key_value[:, self.layer_idx, 1]
```

**复现命令 (需要重新导出 ONNX):**
```bash
bash run_rope_optimize.sh --version v3_kvcache_noslice \
    --modeling_file modeling_qwen2_v3_kvcache_noslice.py \
    --change_node change_node_v3_kvcache_noslice.py \
    --device_id 5
```

**效果分析:**
- 张量 shape 变化: `[1, kv_len, 112, 128]` → permute+view → `[1, 28, 2, 2, kv_len, 128]`
- StridedSliceD: 37.61ms (v2) → **完全消除** (v3), 节省 37.61ms
- 代价: GatherV2 从 8.54ms (2465次) 增加到 16.32ms (4089次), 增加约 8ms
- 净收益: 算子总耗时从 241.13ms 降至 202.92ms, 减少 38.21ms (**-15.8%**)
- 端到端: Decode 速度 112.1 → 131.3 tok/s (**+17.1%**); TTFT 150.0 → 129.8ms (**-13.5%**)
- ONNX 节点数: 8673 (baseline) → 7888 (v3), 减少 785 节点

### v4_noexpand（GQA Broadcast 消除 Expand + 继承 v3 全部优化）

**使用文件:**
- modeling: `export/modeling_qwen2_v4_noexpand.py`
- change_node: `export/change_node_v4_noexpand.py`

**做法:**

1. **GQA Broadcast 替代 repeat_kv** (modeling 层面):
   - 原来: `repeat_kv` 将 K/V 从 `[1, 2, kv_len, 128]` expand 为 `[1, 12, kv_len, 128]`, 产生 Expand 算子
   - 现在: Q reshape 为 `[1, 2, 6, q_len, 128]` (分组视图), K/V unsqueeze 为 `[1, 2, 1, kv_len, 128]`
   - MatMul 自动 broadcast: `[1,2,6,q,128] × [1,2,1,128,kv] → [1,2,6,q,kv]`
   - 完全消除 Expand 算子, 不产生任何额外数据拷贝

2. **继承 v3 全部优化**: 6D KV Cache (消除 StridedSliceD) + RoPE 融合

**关键代码改动 (`modeling_qwen2_v4_noexpand.py`):**

```python
# 原来 (repeat_kv + 标准 matmul):
key_states = repeat_kv(key_states, self.num_key_value_groups)      # expand [1,2,kv,128]→[1,12,kv,128]
value_states = repeat_kv(value_states, self.num_key_value_groups)
attn_weights = torch.matmul(Q / sqrt(d), key_states.transpose(2,3))

# 现在 (grouped broadcast):
query_states = query_states.view(bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, self.head_dim)
key_states = key_states.unsqueeze(2)        # [1, 2, 1, kv_len, 128]
value_states = value_states.unsqueeze(2)    # [1, 2, 1, kv_len, 128]
attn_weights = torch.matmul(Q / sqrt(d), key_states.transpose(3, 4))  # broadcast
attn_output = torch.matmul(attn_weights, value_states)
attn_output = attn_output.view(bsz, self.num_heads, q_len, self.head_dim)  # reshape back
```

**复现命令:**
```bash
bash run_rope_optimize.sh --version v4_noexpand \
    --modeling_file modeling_qwen2_v4_noexpand.py \
    --change_node change_node_v4_noexpand.py \
    --device_id 5
```

**效果分析:**
- Expand: 11.61ms (v3) → **完全消除** (v4)
- ONNX 节点数: 7888 (v3) → 6096 (v4), 减少 1792 节点
- 算子总耗时: 202.92ms (v3) → 182.48ms (v4), 又减少 20.44ms (**-10.1%**)
- 端到端: Decode 131.3 → 140.7 tok/s (**+7.2%**); TTFT 129.8 → 122.4ms (**-5.7%**)
- 相比 baseline 累计: Decode 112.1 → 140.7 tok/s (**+25.5%**); TTFT 150.0 → 122.4ms (**-18.4%**)

## 目录结构

```
opt_models/<version>/          # OM 模型 + 中间 ONNX 文件
  ├── onnx_raw/                # PyTorch 直接导出的 ONNX
  ├── onnx_changed/            # change_node 处理后的 ONNX
  └── *.om                     # 编译后的 OM 模型

opt_profiling/<version>/       # Profiling 数据
  ├── raw_<timestamp>/         # msprof 原始数据 + 解析 CSV
  └── analysis_*.txt           # 算子统计分析报告

opt_benchmark/<version>/       # Benchmark 结果
  └── benchmark_*.txt          # TTFT/TPOT/吞吐量

opt_logs/<version>/            # 日志
  ├── export_onnx_*.log        # ONNX 导出日志
  ├── change_node_*.log        # change_node 日志
  ├── onnx2om_*.log            # ATC 编译日志
  └── node_info_*.txt          # ONNX 节点统计
```

## 剩余瓶颈 (v4_noexpand)

| 算子 | 耗时 | 占比 | 来源 |
|------|------|------|------|
| BatchMatMulV2 | 138.29ms | 75.8% | QKV投影 + Attention + FFN, 主体计算 |
| GatherV2 | 15.68ms | 8.6% | per-layer KV cache 索引 (4089次) |
| ConcatD | 7.39ms | 4.1% | KV cache 拼接 (cat(cache, new_kv), dim=seq) |
| RotaryPositionEmbedding | 5.90ms | 3.2% | 融合后的 RoPE 算子 |
| Add | 5.69ms | 3.1% | 残差连接 |
| Transpose | 3.86ms | 2.1% | 注意力头维度变换 (29次, 单次 133us) |
| SoftmaxV2 | 2.09ms | 1.1% | Attention softmax |

### 可能的进一步优化方向

- **BatchMatMulV2 (75.8%)**: 已占绝对主导, 属于 compute-bound。可考虑量化 (INT8/INT4) 或 FlashAttention 融合来降低
- **GatherV2 (8.6%)**: 6D reshape 引入的索引开销, 4089次/29步≈141次/步。优化空间有限
- **ConcatD (3.6%)**: KV cache seq 维拼接, 不改 OM 协议无法消除
- **Transpose (2.1%)**: 29 次调用单次 133us, 可能通过调整 layout 避免
