# DeepSeek-R1-Distill-Qwen-1.5B 推理性能基准测试报告

> 测试日期: 2026-08-31
> 设备: Atlas 200I A2 (310B1), 1× AI Core, 20T_1.6GHz
> 模型: DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om (FP16, 3.32GB)
> KV Cache 长度: 4096, Prefill 分块: 1 (逐 token)

## 1. 硬件与模型参数

### 1.1 硬件规格

| 项目 | 值 |
|------|-----|
| NPU 芯片 | Ascend 310B1, 1× AI Core |
| FP16 算力 | 10 TFLOPS |
| INT8 算力 | 20 TOPS |
| 内存 | 12 GB LPDDR4X (CPU/NPU 共享) |
| 内存带宽 | 51.2 GB/s (96-bit @ 2131 MHz) |
| 推理可用内存 | ~7.7 GB (扣除系统开销) |

### 1.2 模型架构

| 参数 | 值 |
|------|-----|
| 架构 | Qwen2ForCausalLM |
| 总参数量 | 1.777B |
| FP16 权重大小 | 3.310 GB |
| hidden_size | 1536 |
| intermediate_size | 8960 |
| num_hidden_layers | 28 |
| num_attention_heads | 12 |
| num_key_value_heads | 2 (GQA) |
| head_dim | 128 |
| vocab_size | 151936 |

### 1.3 参数量明细

| 模块 | 每层参数量 | 28 层合计 | FP16 大小 |
|------|----------|----------|----------|
| Q proj (1536→1536) | 2,359,296 | 66.1M | 126.0 MB |
| K proj (1536→256) | 393,216 | 11.0M | 21.0 MB |
| V proj (1536→256) | 393,216 | 11.0M | 21.0 MB |
| O proj (1536→1536) | 2,359,296 | 66.1M | 126.0 MB |
| gate_proj (1536→8960) | 13,762,560 | 385.4M | 734.2 MB |
| up_proj (1536→8960) | 13,762,560 | 385.4M | 734.2 MB |
| down_proj (8960→1536) | 13,762,560 | 385.4M | 734.2 MB |
| Embedding | - | 233.4M | 445.1 MB |
| LM Head | - | 233.4M | 445.1 MB |

## 2. 理论上限分析 (Roofline)

### 2.1 Roofline 模型

性能受限于两种瓶颈中的较大者。判断依据是**算术强度**（Arithmetic Intensity）：

```
算术强度 = 计算量 (FLOPs) / 数据搬运量 (Bytes)
Machine Balance Point = 算力 / 带宽 = 10 TFLOPS / 51.2 GB/s = 195.3 FLOPs/Byte
```

- 算术强度 < Balance Point → **带宽瓶颈** → 性能上限 = 带宽 × 算术强度
- 算术强度 > Balance Point → **计算瓶颈** → 性能上限 = 算力

### 2.2 Decode 阶段 (batch=1, seq_len=1)

每生成一个 token，需读取全部模型权重一次：

```
每 token FLOPs  = 2 × 1.777B = 3.55 GFLOPs
每 token 读取量 = 3.310 GB (权重) + KV Cache
算术强度       = 3.55G / 3.31G ≈ 1.0 FLOPs/Byte << 195.3
→ 带宽瓶颈
```

**理论上限**：

| 序列长度 | 总读取量 | KV Cache 占比 | 理论最大吞吐 | 理论最小 TPOT |
|---------|---------|-------------|------------|-------------|
| 32 | 3.311 GB | 0.0% | 15.5 tok/s | 64.7 ms |
| 100 | 3.313 GB | 0.1% | 15.5 tok/s | 64.7 ms |
| 512 | 3.324 GB | 0.4% | 15.4 tok/s | 64.9 ms |
| 1024 | 3.337 GB | 0.8% | 15.3 tok/s | 65.2 ms |
| 4096 | 3.419 GB | 3.2% | 15.0 tok/s | 66.8 ms |

> KV Cache 每 token 每层: 2(K+V) × 2(heads) × 128(dim) × 2(FP16) = 1024 Bytes
> 28 层合计: 28 KB/token

### 2.3 Prefill 阶段

**批量 Prefill**（一次处理全部 prompt tokens，需要 OM 支持动态 prefill）：

| Prompt 长度 | 计算时间 | 带宽时间 | 瓶颈 | 理论最小 TTFT |
|------------|---------|---------|------|-------------|
| 7 | 2.5 ms | 64.6 ms | 带宽 | 64.6 ms |
| 30 | 10.7 ms | 64.6 ms | 带宽 | 64.6 ms |
| 100 | 35.7 ms | 64.6 ms | 带宽 | 64.6 ms |
| 506 | 184.2 ms | 64.6 ms | **计算** | 184.2 ms |

**逐 token Prefill**（当前 `prefill_length=1`，每个 prompt token 需完整执行一次模型）：

```
理论最小 TTFT = prompt_tokens × 理论最小 TPOT (64.6 ms)
```

| Prompt 长度 | 理论最小 TTFT | 实测 TTFT | 效率 |
|------------|-------------|----------|------|
| 7 tok | 453 ms | 1,457 ms | 31.1% |
| 30 tok | 1,939 ms | 6,013 ms | 32.2% |
| 506 tok | 32,712 ms | 104,812 ms | 31.2% |

## 3. 实测性能数据

### 3.1 Decode 吞吐 (Greedy, prefill_length=1)

| 测试 | Prompt Tokens | 生成 Tokens | TTFT (ms) | TPOT (ms) | Decode (tok/s) |
|------|-------------|------------|-----------|-----------|----------------|
| decode_50 | 7 | 26 (EOS) | 1,464 | 216.8 | 4.6 |
| decode_100 | 7 | 26 (EOS) | 1,458 | 216.4 | 4.6 |
| decode_256 | 7 | 26 (EOS) | 1,457 | 216.4 | 4.6 |

> 模型在 26 token 时自然生成 EOS，decode 速度与目标输出长度无关。

### 3.2 TTFT vs 输入长度 (prefill_length=1)

| 输入 | Prompt Tokens | TTFT (ms) | 每 token Prefill (ms) | Decode (tok/s) |
|------|-------------|-----------|----------------------|----------------|
| 短 | 7 | 1,456 | 208 | 4.6 |
| 中 | 30 | 6,221 | 207 | 4.8 |
| 长 | 506 | 104,812 | 207 | 4.8 |

TTFT 与 prompt 长度严格线性（~207 ms/token），因为 `prefill_length=1` 下每个 prompt token 需一次完整的模型执行。

### 3.3 Prefill 分块长度测试

当前 `_noexpand` 版本 OM 模型**仅支持 `prefill_length=1`**。`prefill_length=2` 时报错：

```
return code is 145012, detail: set_iniput_dynamic_dims
```

这是 TTFT 高的根本原因——无法批量处理 prompt tokens。

### 3.4 采样策略对比 (30 token prompt, 100 gen tokens)

| 采样方式 | TTFT (ms) | TPOT (ms) | Decode (tok/s) | 额外开销 |
|----------|-----------|-----------|----------------|---------|
| Greedy (CPU argmax) | 6,010 | 201.1 | **5.0** | 基线 |
| Top-p=0.8 (CPU) | 6,014 | 205.2 | 4.9 | +2.0% |
| Top-p=0.95 (CPU) | 6,013 | 205.1 | 4.9 | +2.0% |
| Top-k=50 (CPU) | 6,013 | 205.0 | 4.9 | +1.9% |

CPU 采样开销极小（~4 ms/token），不是优化重点。

## 4. Profiling 分析

### 4.1 采集条件

- 工具: msprof (CANN 9.0.0)
- 采集选项: `--aic-metrics=PipeUtilization --aic-freq=100`
- 采集范围: 35 次 ModelExecute（含 prefill + decode）
- 数据路径: `profiling/decode_profile/`

### 4.2 每 token Device 耗时分解 (200.55 ms)

| 算子 | 耗时 (ms) | 占比 | Core 类型 | 说明 |
|------|----------|------|----------|------|
| BatchMatMulV2 | 164.84 | 82.2% | AI_CORE | 线性层 + Attention |
| GatherV2 | 14.38 | 7.2% | AI_VECTOR_CORE | KV Cache 读取 + RoPE |
| Transpose | 11.35 | 5.7% | AI_VECTOR_CORE | KV Cache reshape |
| ConcatD | 6.06 | 3.0% | AI_VECTOR_CORE | KV Cache 拼接 |
| Add | 1.86 | 0.9% | AI_VECTOR_CORE | 残差连接 |
| TransData | 1.05 | 0.5% | AI_VECTOR_CORE | 格式转换 |
| SoftmaxV2 | 0.32 | 0.2% | AI_VECTOR_CORE | Attention softmax |
| 其他 | 0.69 | 0.3% | - | Slice/Neg/Cast 等 |

### 4.3 BatchMatMulV2 细分 (164.84 ms/token)

| 子模块 | 耗时 (ms) | 占 BMM 比 | 占总比 | 说明 |
|--------|----------|----------|-------|------|
| gate_proj | 41.93 | 25.4% | 20.9% | MLP, 1536→8960 |
| up_proj | 42.02 | 25.5% | 21.0% | MLP, 1536→8960 |
| down_proj | 28.10 | 17.0% | 14.0% | MLP, 8960→1536 |
| **MLP 合计** | **112.05** | **67.9%** | **55.9%** | |
| attn_score (QK/AV) | 37.08 | 22.5% | 18.5% | Attention 矩阵乘 |
| o_proj | 7.60 | 4.6% | 3.8% | 1536→1536 |
| q_proj | 6.20 | 3.8% | 3.1% | 1536→1536 |
| k_proj + v_proj | 1.91 | 1.2% | 1.0% | 1536→256 (GQA) |
| **Attention 合计** | **52.79** | **32.1%** | **26.4%** | |

### 4.4 KV Cache 操作分析 (31.67 ms/token, 15.8%)

| 算子 | 操作 | Shape | 耗时 (ms) |
|------|------|-------|----------|
| GatherV2 | 从 KV buffer 按索引读取 | `[1,28,2,2,4096,128]` → 分层 | 14.26 |
| Transpose | KV 维度重排 | `[1,4096,112,128]` → `[1,112,4096,128]` | 11.35 |
| ConcatD | 新 KV 拼接到已有 Cache | `[1,2,4096,128]+[1,2,1,128]` | 5.70 |
| GatherV2 (RoPE) | RoPE 位置编码查表 | `[4097,128]` → 索引 | 0.12 |
| ConcatD (其他) | RoPE concat | `[1,2,1,64]+[1,2,1,64]` 等 | 0.24 |

> Transpose 的 mte2_ratio=0.55，说明完全是内存搬运操作，每次搬运约 112 MB KV 数据。

### 4.5 硬件利用率

| 指标 | 值 | 说明 |
|------|-----|------|
| 实际 TPOT | 201 ms | 实测（Greedy, 30 tok prompt） |
| 理论 TPOT | 64.6 ms | 带宽上限 |
| **带宽效率** | **32.2%** | 实际 / 理论 |
| FP16 算力利用率 | 0.1% | batch=1 矩阵太小 |
| 实际带宽 | ~14.8 GB/s | 3.31GB / 201ms × 1000 (估算) |
| 峰值带宽 | 51.2 GB/s | LPDDR4X |

## 5. 效率差距根因分析

实测 TPOT 201 ms vs 理论 64.6 ms，差距 136.4 ms，效率仅 32.2%。

| 差距来源 | 估算开销 | 占差距比 | 说明 |
|---------|---------|---------|------|
| BMM 带宽利用不足 | ~100 ms | 73% | batch=1 下矩阵形状太小，AI Core 启动开销大，DRAM 访问模式不连续 |
| KV Cache 冗余搬运 | ~32 ms | 23% | GatherV2+Transpose+ConcatD 是纯数据搬运开销 |
| 其他算子 | ~4 ms | 3% | Add/TransData/Softmax/Slice 等 |

## 6. 优化方向与预期收益

### 优化方向 1: 支持批量 Prefill (降低 TTFT)

- **当前问题**: OM 模型仅支持 `prefill_length=1`，30 token prompt 的 TTFT = 6s
- **方案**: 重新导出支持动态 prefill 形状的 OM 模型
- **预期收益**: 30 token prompt TTFT 从 6,013 ms → ~200-400 ms (15-30× 提升)
- **影响范围**: 仅影响 TTFT，不影响 Decode 速度
- **优先级**: **高** — 用户体验影响最大

### 优化方向 2: KV Cache 布局优化 (降低 Decode TPOT)

- **当前问题**: 每 token 花 31.67 ms (15.8%) 在 KV Cache 搬运上
- **根因**: KV Cache 存储布局与模型计算布局不一致，需要 GatherV2+Transpose+ConcatD 三步转换
- **方案**: 在 ONNX 导出阶段调整 KV Cache 的存储布局，使其与计算布局一致，消除运行时搬运
- **预期收益**: TPOT 从 201 ms → ~170 ms，Decode 速度从 5.0 → ~5.9 tok/s
- **优先级**: **中** — 稳定提升 ~18%

### 优化方向 3: 提升 BMM 带宽利用率

- **当前问题**: BMM 耗时 164.84 ms，理论只需 ~64.6 ms（读权重时间）
- **根因**: batch=1 时矩阵 shape 为 `[1, N]×[N, M]`，太小无法充分利用 AI Core
- **方案**:
  - 权重预排布 (weight pre-packing) 适配 AI Core 数据流
  - 算子融合（如 QKV 合并投影、Gate+Up 合并）
  - INT8 量化（权重大小减半，带宽需求减半）
- **预期收益**: 取决于具体方案，INT8 量化理论可将 TPOT 降至 ~100 ms (10 tok/s)
- **优先级**: **中-低** — 需要较大改动，但潜在收益最大

### 收益总结

| 方案 | 目标指标 | 当前值 | 预期值 | 提升幅度 |
|------|---------|--------|-------|---------|
| 批量 Prefill | TTFT (30 tok) | 6,013 ms | ~300 ms | **20×** |
| KV Cache 布局 | Decode TPOT | 201 ms | ~170 ms | 18% |
| INT8 量化 | Decode TPOT | 201 ms | ~100 ms | 2× |
| 全部组合 | Decode TPOT | 201 ms | ~85 ms | 2.4× |

## 7. Profiling 原始数据

- 采集数据: `profiling/decode_profile/PROF_000001_20260831191419866_04143416HCCKEABK/`
- 导出 CSV: `profiling/decode_profile/PROF_000001_20260831191419866_04143416HCCKEABK/mindstudio_profiler_output/`
- 可用 MindStudio 或 Perfetto (chrome://tracing) 打开 timeline 数据
