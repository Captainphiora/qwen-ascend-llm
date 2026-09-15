# 优化记录：v0 ~ v12

> 模型: DeepSeek-R1-Distill-Qwen-1.5B
> 平台: Atlas 200I A2 (310B1) / Atlas 800T A3 (910)
> KV Cache: 4096, Max Prefill: 1, Batch: 1

---

## 性能总览

### 910 (kernel-time, Device 5)

| 版本 | 算子总耗时 | TPOT | 吞吐 | vs baseline | 核心改动 |
|------|-----------|------|------|-------------|---------|
| v0 baseline | 241.13ms | 8.92ms | 112 tok/s | — | 原始 HF 导出 |
| v1 rope | 233.85ms | 8.85ms | 113 tok/s | +0.7% | RoPE change_node 融合 |
| v2 kvcache | 225.46ms | 8.57ms | 117 tok/s | +4.0% | KV cache 静态分配 |
| v3 noslice | 202.92ms | 7.62ms | 131 tok/s | +17.1% | 去 StridedSlice |
| v4 noexpand | 182.48ms | 7.11ms | 141 tok/s | +25.5% | GQA broadcast |

### 910 (kernel-time, Device 1)

| 版本 | TPOT | 吞吐 | 核心改动 |
|------|------|------|---------|
| v5 gate_up_fuse | 6.15ms | 163 tok/s | gate+up concat 融合 |
| v6 transpose_elim | 6.27ms | 160 tok/s | BHSD layout |
| v7 perlayer_kv | 15.53ms | 64 tok/s | 31 输入 per-layer KV (回退) |
| v7b split_kv | 8.76ms | 114 tok/s | 图内 Split (回退) |
| v8 qkv_fuse | 13.07ms | 77 tok/s | QKV 合并 (回退) |
| v11 kv_inplace | 6.05ms | 165 tok/s | KV in-place (Where) |
| v11+QKV | 5.48ms | 183 tok/s | + change_node QKV 合并 |

### 910 (wall-clock, Device 15)

| 版本 | Layout | 量化 | QKV | TPOT | 吞吐 |
|------|--------|------|-----|------|------|
| v5 FP16 | BSHD | — | — | 11.40ms | 88 tok/s |
| v6 FP16 | BHSD | — | — | 14.10ms | 71 tok/s |
| v5+QKV FP16 | BSHD | — | Yes | 10.71ms | 93 tok/s |
| v10 W8A8 | BSHD | W8A8 | — | 10.28ms | 97 tok/s |
| **v5+QKV W8A8** | **BSHD** | **W8A8** | **Yes** | **9.87ms** | **101 tok/s** |

---

## 版本改动对照表

| 版本 | 改了 modeling | 改了 change_node | 需要改推理引擎 |
|------|:---:|:---:|:---:|
| v0 | — | Trilu 修复 | — |
| v1 | | + RoPE 融合 | |
| v2 | ✓ KV cache 静态分配 | | |
| v3 | ✓ KV cache 去 slice | | |
| v4 | ✓ GQA broadcast | | |
| v5 | ✓ gate+up concat 融合 | | |
| v6 | ✓ BHSD layout | | |
| v7 | ✓ per-layer KV | | |
| v7b | ✓ 图内 Split | | |
| v8 | ✓ QKV 合并 (fuse_qkv_weights) | | |
| v9 | ✓ KV slice 优化 | | |
| **v10** | **✓ gate_up 预拼接** | | |
| **v11** | **✓ KV in-place (Where)** | **+ QKV 合并 (--skip_rope)** | **✓ kvcache 需适配** |

> 所有版本的 change_node 都包含 Trilu 修复（310B1 兼容），表中 "+" 表示在此基础上新增的改动。
>
> **当前使用**: v10 modeling + v4 change_node（仅 Trilu 修复）。v11 的 QKV 合并可通过 `change_node_v11 --skip_rope` 单独使用，不需要 v11 的 modeling。

---

## 各版本详解

### v0: baseline

原始 HuggingFace 模型导出 ONNX。KV cache 作为统一张量 `[1, kv_len, num_layers*2*num_kv_heads, head_dim]`，
每层通过 StridedSlice 切片获取 K/V。RoPE 保持原始 `Slice+Neg+Concat+Mul+Add` 模式。

- modeling: `modeling_qwen2.py` (原始 HF)
- change_node: `change_node.py` (仅 Trilu 修复)
- ONNX 节点: 8673, 25 类算子

### v1: RoPE 融合

在 change_node 中匹配 RoPE 子图模式，替换为 `NPURotaryPositionEmbedding` 融合算子。
每层 Q 和 K 各一个，共 56 个 pattern 被替换。

```
Before (7 算子):  Slice → Neg → Concat → Mul(cos) + Mul(sin) → Add
After  (1 算子):  NPURotaryPositionEmbedding(x, cos, sin)
```

- change_node: `change_node_v1_rope.py`
- 节点: 8673 → 8337 (减 336)
- 效果: +0.7% (提升不大，RoPE 本身占比小)

### v2: KV cache 静态分配

将 KV cache 从动态 list 改为预分配的固定张量，在 Python 层面用索引更新替代 cat 拼接。

- modeling: `modeling_qwen2_v2_kvcache.py`
- KV cache shape: `[1, kv_len, num_layers*2*num_kv_heads, head_dim]`
- 效果: +3.3% (减少了 Python 侧 concat 开销)

### v3: 去 StridedSlice

KV cache 切片从 per-iteration 的 `[:, :seq_len]` 改为由 KVCacheManager 管理，
ONNX 图中不再需要 StridedSliceD 切片。

- modeling: `modeling_qwen2_v3_kvcache_noslice.py`
- 节点: 8336 → 7888
- 效果: +12.6% (**消除 StridedSliceD 48.94ms → 0**，最大单版本提升)

### v4: GQA broadcast

GQA 的 K/V repeat 从 `expand+reshape` 改为 `unsqueeze+broadcast`，消除 Expand 算子。

```python
# Before: repeat_kv (产生 Expand 算子)
key_states = repeat_kv(key_states, num_groups)

# After: 5D broadcast (无额外算子)
query_states = query_states.view(bsz, num_kv_heads, num_groups, q_len, head_dim)
key_states = key_states.unsqueeze(2)  # broadcast 自动扩展
```

- modeling: `modeling_qwen2_v4_noexpand.py`
- 节点: 7888 → 6096
- 效果: +7.2% (消除 Expand 11.54ms)

### v5: gate_up concat 融合

MLP 中 gate_proj 和 up_proj 的两次 MatMul 合并为一次:

```python
# Before: 两次 MatMul
gate = F.linear(x, gate_proj.weight)
up = F.linear(x, up_proj.weight)

# After: 一次 MatMul + split
gate_up = F.linear(x, torch.cat([gate_proj.weight, up_proj.weight], dim=0))
gate, up = gate_up.split(intermediate_size, dim=-1)
```

- modeling: `modeling_qwen2_v5_gate_up_fuse.py`
- 效果: 310B1 上 +11.2% (6.38ms, 157 tok/s); **FP16 推荐版本**

### v6: BHSD layout

KV cache layout 从 BSHD `[batch, seq, heads, dim]` 改为 BHSD `[batch, heads, seq, dim]`，
消除 attention 计算前的 Transpose。

- modeling: `modeling_qwen2_v6_transpose_elim.py`
- 效果: kernel-time 略优 (6.27ms vs 6.38ms)，但 wall-clock 下 **BSHD 反而更快**
  (11.40ms vs 14.10ms)，原因不明。**不推荐使用**。

### v7: per-layer KV cache (31 输入)

将统一 KV cache 拆为 28 层独立张量，从 4 输入变为 31 输入。目标是减少 KV cache 搬运量。

- modeling: `modeling_qwen2_v7_perlayer_kv.py`
- 搬运量: 7.98 GB → 4.34 GB (**-45%**)
- 效果: **BMM 退化 2.2 倍**，吞吐从 163 → 64 tok/s。
  31 输入超出 ATC 编译器的 sweet spot，触发图优化退化。**回退方案**。

### v7b: 图内 Split

保持 4 输入接口，在 ONNX 图内用 `torch.chunk(28, dim=1)` 切分 KV cache。

- 效果: BMM 不退化 (4.55ms)，但 Split/Slice 引入 3.33ms 额外开销。
  吞吐 114 tok/s，不如 v5 的 163 tok/s。**回退方案**。

### v8: QKV 投影合并 (modeling 层面)

在 modeling 中定义 `qkv_proj` 线性层，加载权重时通过 `fuse_qkv_weights()` 把 q/k/v 权重
concat 为一个大矩阵。ONNX 导出时图里直接是一个大 MatMul + slice 拆分。

- modeling: `modeling_qwen2_v8_qkv_fuse.py`
- 效果: **BMM 退化 2.0 倍**，吞吐 77 tok/s。
  虽然是在 modeling 层做的合并，但导出后的 ONNX 图拓扑变化仍然触发了 ATC 退化。**回退方案**。

> 教训: v7/v8 的退化都源于 ATC 编译器对图拓扑的敏感性。
> 节点数在 6122 附近是 sweet spot，增加 500+ 节点可能触发退化。
> 退化与 input/output 数量更相关，而非节点数本身。

### v9: KV cache slice 优化

尝试优化 KV cache 的 slice 策略。

- modeling: `modeling_qwen2_v9_kv_slice.py`
- 效果: 未显著改善

### v10: gate_up 权重预拼接 + W8A8

在 PyTorch 层面将 gate_proj 和 up_proj 的权重预拼接为 `gate_up_weight` initializer，
使 AMCT 能正确将其预量化为 INT8 常量。

```python
# v5: 运行时 concat (AMCT 判定为动态 tensor，无法预量化)
gate_up = F.linear(x, torch.cat([gate_proj.weight, up_proj.weight]))

# v10: 预拼接 (单个 initializer，AMCT 正确预量化)
self.gate_up_weight = nn.Parameter(torch.cat([gate_proj.weight, up_proj.weight]))
gate_up = F.linear(x, self.gate_up_weight)
```

- modeling: `modeling_qwen2_v10_gate_up_prefuse.py`
- 效果: 10.28ms, 97 tok/s (wall-clock)。**W8A8 推荐版本**。

### v11: KV in-place + QKV 合并

KV cache 更新从 `ScatterElements`（无 NPU kernel，落 AI_CPU）改为 `Where(SelectV2)`，
同时在 change_node 中做 QKV 投影合并（与 v8 不同，v11 不改变原图拓扑，BMM 不退化）。

- modeling: `modeling_qwen2_v11_kv_inplace.py`
- change_node: `change_node_v11_kv_inplace.py`
- v11 单独: 6.05ms, 165 tok/s
- v11+QKV: 5.48ms, **183 tok/s (FP16 最优)**
- v11 W8A8: 12.80ms, 78 tok/s (BHSD 图上 AMCT 的 Vector 退化，不推荐)

### v12: v10 W8A8 + v11 QKV 合并

v10 的 W8A8 量化 + v11 的 QKV 合并，在 AMCT 校准前做 QKV merge（`--skip_rope`），
保持标准 ONNX 算子兼容 AMCT。

- 效果: 9.87ms, 101 tok/s (wall-clock)。**W8A8 + QKV 最优**。

---

## 关键算子演进 (310B1, v0 → v4)

| 算子 | v0 | v1 | v2 | v3 | v4 | 变化 |
|------|-----|-----|-----|-----|-----|------|
| BatchMatMulV2 | 138.8ms | 142.7ms | 138.6ms | 145.5ms | 138.3ms | 不变 (75.8%) |
| StridedSliceD | 48.9ms | 41.8ms | 37.6ms | **消除** | **消除** | v3 消除 |
| ConcatD | 20.4ms | 13.8ms | 7.3ms | 7.3ms | 7.4ms | v1/v2 减半 |
| Expand | 11.5ms | 11.0ms | 11.6ms | 11.6ms | **消除** | v4 消除 |
| GatherV2 | 2.5ms | 2.5ms | 8.5ms | 16.3ms | 15.7ms | v2/v3 增加 |
| RoPE | — | 6.2ms | 6.0ms | 6.0ms | 5.9ms | v1 新增 |

> BMM 始终占 57%~76%，是绝对主体。优化的本质是**消除 BMM 以外的冗余算子**。

---

## 搬运量分析

### v5 每 token 搬运量拆解

| 类别 | 搬运量 | 占比 |
|------|--------|------|
| 权重 (W) | 3.087 GB | 39% |
| KV cache | 4.753 GB | 60% |
| 激活 | 0.025 GB | 1% |
| **总计** | **7.984 GB** | |

> 注意: KV cache 声明搬运量 4.75 GB，但实测仅占 0.81ms/tok (13%)。
> 声明搬运量 ≠ 实际 HBM 读取量，GatherV2 等算子的实际读取远小于声明值。
> **真正的瓶颈是 BMM (74%)**，不是 KV cache。

### v7 搬运量对比

| 版本 | 总搬运(声明) | 实测吞吐 | 有效带宽 |
|------|------------|---------|---------|
| v5 | 7.98 GB | 163 tok/s | 1298 GB/s |
| v7 (31输入) | 4.34 GB | 64 tok/s | 280 GB/s |
| v7b (4输入Split) | 7.75 GB | 114 tok/s | 884 GB/s |

搬运量减少 45% 但吞吐反降 60%，证实了声明搬运量的误导性。

---

## vLLM-Ascend 对比

| 框架 | TPOT | 吞吐 | TTFT |
|------|------|------|------|
| vLLM V0+Eager | 27.62ms | 36 tok/s | 77.7ms |
| vLLM V1+Eager | 27.71ms | 36 tok/s | 69.6ms |
| vLLM V1+ACL Graph | 16.01ms | 63 tok/s | 53.2ms |
| OM v0 baseline | 8.92ms | 112 tok/s | 150.0ms |

OM 推理 decode 吞吐为 vLLM 的 1.8~3.1 倍，但 TTFT (首 token 延迟) 更高。
vLLM 在长序列、动态 batch、多并发场景下有优势。

---

## 关键经验

1. **profiling 覆盖范围**: msprof 能采集 NPU 算子（`op_summary.csv` 中的 `Task Duration` 含调度）和 host 侧 ACL API 调用（`api_statistic.csv`，需 `--runtime-api=on`，脚本默认开启）。`step_trace.csv` 的 `Iteration Time` 包含 host+device 的完整时间。唯一采不到的是不经过 ACL 的纯 Python 逻辑（tokenizer、numpy 采样等），v5 在 910 上 kernel-time 累计 6.15ms 但 wall-clock 11.40ms，差距主要来自这部分。
2. **BSHD 优于 BHSD**: wall-clock 下 BSHD 快 19%，与 kernel-time 结论相反
3. **ATC 对图拓扑极度敏感**: 节点数 6122 附近是 sweet spot，+500 可能触发 BMM 2 倍退化
4. **声明搬运量 ≠ 实际搬运量**: GatherV2 等算子的实际 HBM 读取远小于声明值
5. **BMM 是绝对瓶颈**: 占 74%~76%，算术强度仅 1.0 FLOPs/Byte，完全 memory-bound
