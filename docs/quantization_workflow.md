# W8A8 量化完整流程

> 模型: DeepSeek-R1-Distill-Qwen-1.5B
> 量化工具: amct_onnx (CANN 9.0.0)
> 目标平台: Atlas 200I A2 (310B1) / Atlas 800T A3 (910)

---

## 流程总览

有两个版本: v10 原始流程和 v12 优化流程 (在 v10 基础上增加 QKV 合并)。

### v10 原始流程

```
PyTorch 模型 (HuggingFace FP16)
    │  Step 1: 导出 ONNX（预拼接 gate_up 权重）
    ▼
raw ONNX (FP16, 标准算子)
    │  Step 2: AMCT 校准量化
    ▼
deploy ONNX (INT8 权重 + AscendQuant/AscendDequant)
    │  Step 3: 精度验证
    │  Step 4: change_node（RoPE 融合 / Trilu 修复）
    ▼
changed ONNX (含昇腾自定义算子)
    │  Step 5: ATC 编译
    ▼
OM 模型 (可部署)
```

### v12 优化流程 (推荐)

在 v10 基础上增加 QKV 投影合并, 910 上实测 ~2.5% 吞吐提升。

```
PyTorch 模型 (HuggingFace FP16)
    │  Step 1: 导出 ONNX（预拼接 gate_up 权重）
    ▼
raw ONNX (FP16, 标准算子)
    │  Step 1.5: QKV 合并 (change_node, 标准 ONNX 算子)    ← 新增
    ▼
QKV-merged ONNX (FP16, 标准算子, q/k/v_proj → qkv_proj)
    │  Step 2: AMCT 校准量化
    ▼
deploy ONNX (INT8 权重 + AscendQuant/AscendDequant)
    │  Step 3: 精度验证
    │  Step 4: change_node（RoPE 融合 / Trilu 修复）       ← v10 原有
    ▼
changed ONNX (含昇腾自定义算子)
    │  Step 5: ATC 编译
    ▼
OM 模型 (可部署)
```

### 性能对比 (910, Device 15, wall-clock, batch=1, kv_cache=4096)

| 版本 | TPOT | 吞吐 |
|------|------|------|
| v5 FP16 baseline | 11.40 ms | 88 tok/s |
| v10 W8A8 (原始流程) | 10.28 ms | 97 tok/s |
| **v12 W8A8+QKV (优化流程)** | **9.87 ms** | **101 tok/s** |

---

## 环境准备

每次新终端都需要执行:

```bash
source /mnt/host-model/cxj/npu_workflow_demo/npu_env.sh qwen_ascend_cann900
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/lib64/plugin/opskernel:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PATH=/usr/local/Ascend/cann-9.0.0/bin:/usr/local/Ascend/cann-9.0.0/tools/profiler/bin:$PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp
export PYTHONPATH=/mnt/host-model/cxj/qwen-ascend-llm:$PYTHONPATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

cd /mnt/host-model/cxj/qwen-ascend-llm
```

后续所有命令均使用 `conda run -n qwen_ascend_cann900` 执行。

---

## Step 1: 导出 ONNX

### 预量化的关键: 预拼接 gate_up 权重

v10 的核心改动在 `export/modeling_qwen2_v10_gate_up_prefuse.py` 中:

- `Qwen2MLP` 新增 `fuse_gate_up()` 方法, 在权重加载后将 `gate_proj.weight` 和 `up_proj.weight` 合并为单个 `gate_up_weight` 参数
- `Qwen2ForCausalLM` 新增 `fuse_gate_up_weights()` 方法, 遍历所有 MLP 层调用 `fuse_gate_up()`
- `export_onnx.py` 在导出前自动检测并调用 `fuse_gate_up_weights()`

**为什么必须这样做**: 原版 v5 的 `F.linear(x, torch.cat([gate_proj.weight, up_proj.weight]))` 在 ONNX 中生成 `Concat` 节点。AMCT 看到 Concat 输出会将其判定为动态 tensor, 无法预量化为 INT8 常量, 导致运行时用 AscendQuant 实时转换（每次 3.1 ms, 累计 3,072 ms）。预拼接后 ONNX 中只有一个 initializer, AMCT 正确预量化。

### 命令

```bash
# 复制 v10 modeling 为 export 使用的文件
cp export/modeling_qwen2_v10_gate_up_prefuse.py export/modeling_qwen2.py

# 导出 ONNX (NPU + FP16)
mkdir -p opt_models/v10_gate_up_prefuse/onnx_raw
conda run -n qwen_ascend_cann900 \
  python3 export/export_onnx.py \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --onnx_model_path opt_models/v10_gate_up_prefuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --kv_cache_length 4096 \
    --kv_cache_layout BSHD \
    --device_str npu \
    --dtype float16

# 还原 modeling_qwen2.py
git checkout export/modeling_qwen2.py
```

输出日志中应看到 `[INFO] Fused gate_up weights in 28 MLP layers`。

### 产出

- `opt_models/v10_gate_up_prefuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx` + 外部数据文件

---

## Step 1.5: QKV 投影合并 (v12 优化流程, 可选)

> 仅 v12 优化流程需要此步骤。v10 原始流程跳过此步直接进入 Step 2。

将 q_proj + k_proj + v_proj 三个 MatMul 合并为一个 qkv_proj MatMul + Split。
在 AMCT 之前做, 让 AMCT 将合并后的 qkv_proj 作为整体量化。

**关键**: 此步只做 QKV 合并, 不做 RoPE 融合 (`--skip_rope`), 保持标准 ONNX 算子以兼容 AMCT。

### 命令

```bash
conda run -n qwen_ascend_cann900 \
  python3 export/change_node_v11_kv_inplace.py \
    --input_model_path opt_models/v10_gate_up_prefuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --output_model_path opt_models/v10_gate_up_prefuse/onnx_qkv/deploy_qkv.onnx \
    --skip_rope
```

输出应看到 `Found 28 QKV groups to merge`, `RoPE fusion skipped`。

### 验证

```bash
# 确认无自定义算子 (AMCT/onnxruntime 兼容)
python3 -c "
import onnx
m = onnx.load('opt_models/v10_gate_up_prefuse/onnx_qkv/deploy_qkv.onnx')
custom = [n.op_type for n in m.graph.node if 'Ascend' in n.op_type or 'NPU' in n.op_type]
assert not custom, f'Found custom ops: {custom}'
print('OK: all standard ONNX ops')
"
```

### 产出

- `opt_models/v10_gate_up_prefuse/onnx_qkv/deploy_qkv.onnx` + 外部数据文件
- MatMul 节点: 169 个 (原 225 个, 减少 56 个 q/k/v_proj, 增加 28 个 qkv_proj)

---

## Step 2: AMCT 校准量化

AMCT (Ascend Model Compression Toolkit) 对 ONNX 做 W8A8 PTQ:

1. `create_quant_config`: 读取 .cfg 文件, 确定哪些层量化/跳过
2. `quantize_model`: 在图中插入校准统计节点
3. 校准推理: 用 50 条 boolq 数据在 CPU 上跑 modified ONNX, 收集激活分布
4. `save_model`: 产出 deploy ONNX (真正 INT8) + fake_quant ONNX (FP16 模拟)

### cfg 文件

`scripts/quant_v8_skip60_gate_up_quantized.cfg` — 跳过 60 层:

| 跳过类型 | 数量 | 原因 |
|---------|------|------|
| lm_head | 1 | 输出层精度敏感 |
| Attention QK^T | 28 | 两端都是激活, 无固定权重 |
| Attention Score×V | 28 | 两端都是激活 |
| outlier down_proj (L2/26/27) | 3 | 激活存在极端离群值 |

gate_up_fuse 和 qkv_proj (v12) 不跳过 — 它们是 activation × weight, 适合 W8A8。
同一份 cfg 文件同时适用于 v10 (225 MatMul) 和 v12 (169 MatMul), 因为 skip 的节点名
(self_attn/MatMul, self_attn/MatMul_1, lm_head/MatMul, down_proj/MatMul) 在两个版本中相同。

### 校准数据

BoolQ 数据集, 50 条完整 prompt prefill, 路径:
`/usr/local/Ascend/atb-models/examples/convert/model_slim/boolq.jsonl`

### 校准输入构造注意事项

校准脚本 (`scripts/amct_onnx_calibrate.py`) 构造 ONNX 输入时:
- `past_key_values`: 用 **kv_len=1** (BSHD dim 1 是 dynamic), 不要用 kv_len=4096
- `attention_mask`: 长度 = seq_len + 1, 值为 `[1...1, 0]` (最后一位为 0)
- 这样模型在校准时做正常的 causal attention, 收集到真实的激活分布

**已知陷阱**: 如果用 kv_len=4096 + mask 前 seq_len 位为 1, 模型会 attend 到全零的 past
KV buffer, 导致校准 scale 偏移, 输出质量严重下降。

### 命令

v10 原始流程:
```bash
conda run -n qwen_ascend_cann900 \
  python3 scripts/amct_onnx_calibrate.py \
    --model_path opt_models/v10_gate_up_prefuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output_dir opt_models/v10_gate_up_prefuse/amct_output \
    --num_samples 50 \
    --kv_cache_length 4096 \
    --cpu_threads 64 \
    --quant_cfg scripts/quant_v8_skip60_gate_up_quantized.cfg
```

v12 优化流程 (用 QKV 合并后的 ONNX):
```bash
conda run -n qwen_ascend_cann900 \
  python3 scripts/amct_onnx_calibrate.py \
    --model_path opt_models/v10_gate_up_prefuse/onnx_qkv/deploy_qkv.onnx \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output_dir opt_models/v10_gate_up_prefuse/amct_qkv_output \
    --num_samples 50 \
    --kv_cache_length 4096 \
    --cpu_threads 64 \
    --kv_cache_layout BSHD \
    --quant_cfg scripts/quant_v8_skip60_gate_up_quantized.cfg
```

耗时约 2-10 分钟（CPU 推理 50 条 prompt）。

### 产出

| 文件 | 用途 |
|------|------|
| `amct_output/model_deploy_deploy_model.onnx` | 部署级 INT8 ONNX, 给 ATC 编译 |
| `amct_output/model_deploy_fake_quant_model.onnx` | FP16 仿真, 给精度验证 |
| `amct_output/record.txt` | 校准 scale 数据 |
| `amct_output/quant_config.json` | 量化配置 |

---

## Step 3: 精度验证

**在编译 OM 之前验证**, 避免浪费编译时间。

```bash
conda run -n qwen_ascend_cann900 \
  python3 scripts/verify_fakequant.py \
    --fp16_onnx opt_models/v10_gate_up_prefuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
    --fakequant_onnx v10=opt_models/v10_gate_up_prefuse/amct_output/model_deploy_fake_quant_model.onnx \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B
```

### 合格标准

| 指标 | 要求 |
|------|------|
| Cosine similarity | > 0.8 |
| Top-1 match | True |
| Top-5 overlap | >= 4/5 |

v10 实测: cosine=0.884, top-1=Y, top-5=5/5, PASS。

---

## Step 4: change_node

对 deploy ONNX 做图改写。注意 gate_up_fuse 已在 Step 1 的 ONNX 导出中完成（modeling 层面），
change_node 只处理 RoPE 和 Trilu，与 gate_up 无关。

根据目标平台选择不同脚本:

| 平台 | 脚本 | 做什么 | 不做什么 |
|------|------|--------|---------|
| 910 | `change_node_v5_gate_up_fuse.py` | RoPE → NPURotaryPositionEmbedding | 名字有误导, 不处理 gate_up |
| 310B1 | `change_node_v4_noexpand_310b.py` | Trilu 修复 | 310B1 不支持 RoPE 融合算子 |

### 310B1

```bash
conda run -n qwen_ascend_cann900 \
  python3 export/change_node_v4_noexpand_310b.py \
    --input_model_path opt_models/v10_gate_up_prefuse/amct_output/model_deploy_deploy_model.onnx \
    --output_model_path opt_models/v10_gate_up_prefuse/deploy_changed_310b.onnx
```

### 910

```bash
conda run -n qwen_ascend_cann900 \
  python3 export/change_node_v5_gate_up_fuse.py \
    --input_model_path opt_models/v10_gate_up_prefuse/amct_output/model_deploy_deploy_model.onnx \
    --output_model_path opt_models/v10_gate_up_prefuse/deploy_changed_910.onnx
```

注意两个脚本的参数名不同（`--input/--output` vs `--input_model_path/--output_model_path`）。

---

## Step 5: ATC 编译 OM

### 310B1

```bash
conda run -n qwen_ascend_cann900 \
  python3 export/onnx2om.py \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --onnx_model_path opt_models/v10_gate_up_prefuse/deploy_changed_310b.onnx \
    --om_model_path opt_models/v10_gate_up_prefuse/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v10_310b \
    --kv_cache_length 4096 \
    --max_prefill_length 1 \
    --kv_cache_layout BSHD \
    --precision_mode origin \
    --soc_version Ascend310B1 \
    --cpu_thread 16
```

### 910

```bash
conda run -n qwen_ascend_cann900 \
  python3 export/onnx2om.py \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --onnx_model_path opt_models/v10_gate_up_prefuse/deploy_changed_910.onnx \
    --om_model_path opt_models/v10_gate_up_prefuse/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v10_910 \
    --kv_cache_length 4096 \
    --max_prefill_length 1 \
    --kv_cache_layout BSHD \
    --precision_mode origin \
    --soc_version Ascend910_9382 \
    --cpu_thread 16
```

关键参数:
- `--precision_mode origin`: AMCT 量化模型必须用 origin, 不能用 mixed_float16
- `--max_prefill_length 1`: 静态 shape, OM 最小

### 产出

| 平台 | OM 路径 | 大小 |
|------|---------|------|
| 310B1 | `opt_models/v10_gate_up_prefuse/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v10_310b.om` | 2.2 GB |
| 910 | `opt_models/v10_gate_up_prefuse/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v10_910.om` | 2.2 GB |

---

## 验证部署

```bash
# Benchmark
conda run -n qwen_ascend_cann900 \
  python3 benchmarks/benchmark.py \
    --om_model_path opt_models/v10_gate_up_prefuse/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v10_310b.om \
    --device_id 0 \
    --max_prefill_length 1 \
    --rounds 5 --warmup 2 --max_new_tokens 50 \
    --label "v10_w8a8_310b"
```

---

## 文件索引

| 文件 | 作用 |
|------|------|
| `export/modeling_qwen2_v10_gate_up_prefuse.py` | v10 建模文件 (预拼接 gate_up 权重) |
| `export/export_onnx.py` | ONNX 导出 (自动检测 fuse_gate_up_weights) |
| `export/change_node_v11_kv_inplace.py` | QKV 合并 + RoPE (v12 用 `--skip_rope` 只做 QKV) |
| `scripts/quant_v8_skip60_gate_up_quantized.cfg` | AMCT 量化配置 (skip 60, v10/v12 通用) |
| `scripts/amct_onnx_calibrate.py` | AMCT 校准脚本 (支持 `--kv_cache_layout`) |
| `scripts/verify_fakequant.py` | 精度验证脚本 |
| `export/change_node_v4_noexpand_310b.py` | 310B1 图改写 |
| `export/change_node_v5_gate_up_fuse.py` | 910 图改写 (RoPE 融合) |
| `export/onnx2om.py` | ATC 编译封装 |
