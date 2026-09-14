# 优化进度记录

> 日期: 2026-09-11
> 模型: DeepSeek-R1-Distill-Qwen-1.5B
> 平台: Atlas 800T A3, Ascend 910_9382
> 测试条件: batch=1, kv_cache_length=4096, decode, Device 15, wall-clock

---

## 1. 当前性能总览 (Device 15, wall-clock)

| 版本 | Layout | 量化 | QKV合并 | TPOT | 吞吐 | 精度 |
|------|--------|------|---------|------|------|------|
| v5 orig FP16 | BSHD | — | — | 11.40 ms | 88 tok/s | ✅ |
| v6 orig FP16 | BHSD | — | — | 14.10 ms | 71 tok/s | ✅ |
| v5+QKV FP16 | BSHD | — | ✅ | 10.71 ms | 93 tok/s | ✅ |
| v6+QKV FP16 (v11) | BHSD | — | ✅ | 13.80 ms | 72 tok/s | ✅ |
| v10 W8A8 (原始) | BSHD | W8A8 | — | 10.28 ms | 97 tok/s | ✅ |
| **v5+QKV W8A8** | **BSHD** | **W8A8** | **✅** | **9.87 ms** | **101 tok/s** | **✅** |

### 结论

- **BSHD 全面优于 BHSD**: v5(11.40) vs v6(14.10), wall-clock 下 BSHD 快 19%
- **v6 原始 BHSD 的 profiling kernel-time 优势在 wall-clock 下不成立**
- **QKV 合并在 BSHD 上有效**: -6.1% (FP16), -4.0% (W8A8)
- **QKV 合并在 BHSD 上无效**: v6+QKV(13.80) vs v6(14.10) 仅 -2.1%
- **W8A8 量化有效**: v5+QKV FP16(10.71) → v5+QKV W8A8(9.87) = -7.8%
- **最优: v5+QKV W8A8 = 9.87 ms (101 tok/s)**

## 2. 已实施的优化

### 2.1 QKV 权重投影合并 (change_node 层面)

**做法**: ONNX 图级别，将 q_proj/k_proj/v_proj 三个 MatMul 合并为一个 qkv_proj MatMul + Split。

```
Before: hidden → MatMul(W_q) → Q     (3 个 kernel, 权重 1536+256+256=2048 维)
        hidden → MatMul(W_k) → K
        hidden → MatMul(W_v) → V

After:  hidden → MatMul(W_qkv) → Split → Q, K, V   (1 个大 kernel + 1 Split)
```

**实现**: `export/change_node_v11_kv_inplace.py --skip_rope`
- 权重: numpy concat W_q[1536,1536] + W_k[256,1536] + W_v[256,1536] → W_qkv[2048,1536]
- 偏置: 同样 concat
- Split sizes 作为 ONNX input (opset 13+ 兼容)

**收益**: 减少 2 个 MatMul kernel launch/token × 28 层 = 56 kernel, 小权重的 launch 开销占比极高

### 2.2 W8A8 AMCT 量化

**做法**: amct_onnx PTQ 量化，skip 60 层 (lm_head + attention matmul + outlier down_proj)。

**流程**:
```
v10 modeling (BSHD, gate_up prefuse)
  → export_onnx.py (FP16)
    → change_node (QKV merge, skip RoPE)
      → amct_onnx_calibrate.py (50 boolq samples, kv_len=1)
        → ATC (--precision_mode origin, BSHD)
```

**关键**: 校准时 past_key_values 用 kv_len=1 (BSHD dim 1 是 dynamic)，
attention_mask 用 `[1...1, 0]` (seq_len+1)。不要用 kv_len=4096。

## 3. 关键经验教训

### 3.1 测试方法

- **必须用 wall-clock 对比**, 不能只看 profiling kernel-time
- kernel-time 不含 H2D/D2H/host 开销, 对 FP16 大权重模型低估实际延时
- profiling (msprof) 本身有 overhead, 在繁忙设备上会放大
- **用同一设备、同一方法测所有版本**

### 3.2 BSHD vs BHSD

- BHSD kernel-time 更短 (ATC 对 BHSD 的 MatMul 调度更紧凑)
- 但 BSHD wall-clock 更短 (权重搬运和 host 交互更高效)
- **W8A8 只在 BSHD 上正常工作** (BHSD 触发 Vector Core 退化)
- 后续统一用 BSHD

### 3.3 AMCT 校准

- kv_len=1 + mask `[1...1, 0]` 是正确做法 (BSHD 的 kv_len 是 dynamic)
- kv_len=4096 + mask `[1...1, 0...0]` 会导致模型 attend 到全零 past → 校准质量崩溃
- AMCT deploy ONNX 结构 (AscendQuant/Dequant 数量) 正确, 问题仅在校准 scale

### 3.4 ATC 编译器敏感性

- input/output 数量不变时, 图拓扑变化 (QKV 合并) 不触发退化
- 量化 (AMCT) 增加的 AscendQuant/Dequant 节点不触发退化 (ATC 自动融合为 QuantBMM)
- 之前观察到的退化 (v7/v8) 可能与 input 数量变化相关

## 4. 文件清单

| 文件 | 说明 |
|------|------|
| `export/change_node_v11_kv_inplace.py` | QKV merge + RoPE (可选), `--skip_rope` `--skip_qkv_merge` |
| `scripts/amct_onnx_calibrate.py` | AMCT 校准 (修复: kv_len=1, 支持 --kv_cache_layout) |
| `scripts/quant_v8_skip60_gate_up_quantized.cfg` | AMCT skip 配置 (60 层) |
| `opt_models/v5_fp16_test/v5_baseline.om` | v5 FP16 baseline |
| `opt_models/v5_fp16_test/v5_qkv.om` | v5+QKV FP16 |
| `opt_models/v5_fp16_test/v5_qkv_w8a8.om` | **v5+QKV W8A8 (当前最优)** |
| `opt_models/v5_fp16_test/deploy_qkv.onnx` | QKV 合并后的 FP16 ONNX |
| `opt_models/v5_fp16_test/amct_qkv/` | AMCT 校准产出 |

### 复现命令

```bash
# 环境
source /mnt/host-model/cxj/npu_workflow_demo/npu_env.sh qwen_ascend_cann900
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
# ... (见 AGENTS.md)

# Step 1: 导出 ONNX (v10 modeling, BSHD)
cp export/modeling_qwen2_v10_gate_up_prefuse.py export/modeling_qwen2.py
python export/export_onnx.py \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --onnx_model_path opt_models/v5_fp16_test/onnx_raw/deploy.onnx \
  --kv_cache_length 4096 --kv_cache_layout BSHD \
  --device_str npu --dtype float16

# Step 2: QKV 合并 (无 RoPE)
python export/change_node_v11_kv_inplace.py \
  --input_model_path opt_models/v5_fp16_test/onnx_raw/deploy.onnx \
  --output_model_path opt_models/v5_fp16_test/deploy_qkv.onnx \
  --skip_rope

# Step 3: AMCT 校准
python scripts/amct_onnx_calibrate.py \
  --model_path opt_models/v5_fp16_test/deploy_qkv.onnx \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --output_dir opt_models/v5_fp16_test/amct_qkv \
  --num_samples 50 --kv_cache_length 4096 --cpu_threads 64 \
  --kv_cache_layout BSHD \
  --quant_cfg scripts/quant_v8_skip60_gate_up_quantized.cfg

# Step 4: ATC 编译
python export/onnx2om.py \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --onnx_model_path opt_models/v5_fp16_test/amct_qkv/model_deploy_deploy_model.onnx \
  --om_model_path opt_models/v5_fp16_test/v5_qkv_w8a8 \
  --kv_cache_length 4096 --max_prefill_length 1 \
  --kv_cache_layout BSHD --precision_mode origin --cpu_thread 16
```

## 5. 后续优化方向

### 5.1 定长 KV Cache 问题

当前每次推理都传入完整 `[1, 4096, 112, 128]` 的 KV buffer (约 115 MB)。
即使只用了前 N 个位置，后面 4096-N 个零位置仍参与 attention 计算:
- GatherV2 每层读取整个 buffer
- Attention MatMul 在 4097 维度上做，包含大量无效零位

**可能方案**:
- **动态 KV 长度**: onnx2om 的 `--dynamic_dims` 支持多档 kv_len (已有 4096/2048 两档)。
  可增加更多档位 (256/512/1024) 减少早期推理的搬运量
- **KV cache 分片**: engine 侧根据 real_kv_size 选择对应档位的 dynamic shape 调用
- **OM 输入 mask**: 当前 mask 已屏蔽无效位置，但 MatMul 仍在全长度上运算。
  需要 ATC 支持 mask-aware 的稀疏注意力才能真正跳过无效计算

### 5.2 enable_compress_weight

ATC 编译选项 `--enable_compress_weight=true`，权重 HBM 压缩存储、读取时硬件解压。
零代码改动，对访存瓶颈模型可能有 10-20% 收益。待验证。

### 5.3 MindIE / vLLM-Ascend

绕过 OM pipeline，直接用框架级优化 (FlashAttention, PagedAttention, continuous batching)。
batch=1 预期 3-5 ms/tok; batch=8+ 预期 <2 ms/tok。

### 5.4 RoPE 融合 (已放弃)

NPURotaryPositionEmbedding 融合在 profiling kernel-time 上有 ~0.1ms 收益，
但增加了 pipeline 复杂度 (AMCT 不兼容自定义 op)。已决定不做。
