# W8A8 量化探索记录 (HANDOFF)

## 目标

在 Ascend 910 (CANN 9.0.0) 上实现 DeepSeek-R1-Distill-Qwen-1.5B 的 INT8 (W8A8) 量化推理加速，通过 ONNX→OM 路径部署。

## 模型来源

- 原始 FP16 模型：`DeepSeek-R1-Distill-Qwen-1.5B`
- W8A8 量化模型：克隆自 [modelers.cn](https://modelers.cn/models/MindIE/DeepSeek-R1-Distill-Qwen-1.5B-OrangePi/tree/main/deepseek-qwen-1.5B-w8a8)
- 量化命令（官方）：
  ```bash
  python3 msit/msmodelslim/example/Qwen/quant_qwen.py \
    --model_path $ORG --save_directory $MODEL \
    --calib_file msit/msmodelslim/example/common/boolq.jsonl \
    --w_bit 8 --a_bit 8 --device_type npu \
    --disable_names "lm_head" --anti_method m4
  ```

---

## kv_dim 和 per_head_dim 的计算

### 本项目中的计算方式

从 HuggingFace 模型 config 读取：

```python
from transformers.models.qwen2 import Qwen2Config
config = Qwen2Config.from_pretrained(hf_model_dir)

num_hidden_layers = config.num_hidden_layers        # 28
num_attention_heads = config.num_attention_heads    # 12
num_key_value_heads = config.num_key_value_heads    # 2
hidden_size = config.hidden_size                    # 1536
per_head_dim = hidden_size // num_attention_heads   # 1536 // 12 = 128
```

### kv_dim 的含义

KV cache 的第三维度，表示所有层的 K 和 V head 打包在一起：

```python
kv_dim = num_hidden_layers * 2 * num_key_value_heads  # 28 * 2 * 2 = 112
```

- `num_hidden_layers * 2`：每层有 Key 和 Value 两组
- `* num_key_value_heads`：GQA (Grouped Query Attention)，每组有 2 个 KV head

### KV cache 完整 shape

```
past_key_values: [batch_size, kv_cache_length, kv_dim, per_head_dim]
                 [1,          4096,            112,    128]
```

### 原始项目中的获取方式

`export/export_onnx.py` 和 `export/onnx2om.py` 都从 config 读取：
```python
test_model_config = Qwen2Config.from_pretrained(args.hf_model_dir)
num_hidden_layers = test_model_config.num_hidden_layers
num_key_value_heads = test_model_config.num_key_value_heads
per_head_dim = test_model_config.hidden_size // test_model_config.num_attention_heads
```

---

## 已完成的工作

### 1. msmodelslim W8A8 模型的 ONNX 导出 ✅

**文件：** `export/quantize/w8a8_linear.py`, `export/export_onnx.py`

- 创建 `W8A8PreQuantizedLinear` 类，直接从 msmodelslim 的 safetensors 加载 int8 权重 + weight_scale
- forward 产出 `Cast(int8→fp16) → Mul(weight_scale) → MatMul` 的 ONNX pattern
- 导出成功验证：196 个量化层，INT8 initializer 正确

**问题：** 导出的 ONNX 经 ATC 编译时，`Cast→Mul` 被常量折叠回 FP16，OM 无实际 INT8 加速。

### 2. tokenizer chat_template 修复 ✅

量化模型的 `tokenizer_config.json` 缺少 `<think>\n` 后缀，导致模型不进入思考模式。已修复。

### 3. AMCT 环境搭建 ✅

- 安装 `Ascend-cann-amct_9.0.0`（amct_acl + amct_onnx）
- 编译 amct_onnx 自定义算子库 `libamct_onnx_ops.so`（绕过 onnxruntime 版本检查，用 1.16.0 headers 编译）
- onnxruntime 1.23.2 + onnxruntime-cann 1.23.2 + numpy<2 环境可用

### 4. amct_onnx PTQ 校准流程 ✅ (CPU 校准可行)

**文件：** `scripts/amct_onnx_calibrate.py`, `scripts/amct_onnx_ptq.sh`

流程：
1. `create_quant_config` → 生成量化配置
2. `quantize_model` → 插入校准节点（内含 TransposeFoldPass）
3. onnxruntime CPU 推理 → 收集激活统计 → record.txt
4. `save_model` → 产出 deploy_model.onnx（权重 INT8 + 量化参数）
5. ATC 编译 → OM

**状态：** 编译通过，但首次推理输出乱码（因为校准数据用了随机 KV cache + 仅 1 个样本）。已修复为全零 KV cache + 多样本，等待重新验证。

### 5. change_node_v4_noexpand_310b ONNX 生成 ✅

无 RotaryMul 自定义算子的 ONNX（纯标准 ONNX ops），用于 AMCT 校准和 ATC 编译。

---

## 尝试过但不可行的路径

### 路径 A: 自定义算子 QuantBatchMatmulV3 ❌

**做法：** change_node_quant.py 将 Cast→Mul→MatMul 替换为 AscendQuant + QuantBatchMatmulV3

**失败原因：** ATC 不认识 QuantBatchMatmulV3 算子（`No operator plugin is registered`），该算子可能只在 MindIE/ATB 框架中可用。

### 路径 B: ATC --compression_optimize_conf (NPU 校准) ❌

**做法：** 生成校准 bin 文件 + cfg，让 ATC 内置 AMCT 在 NPU 上做校准推理和量化编译。

**失败过程：**
1. CANN 9.0.0 无 AMCT 库 → 安装 amct_acl 9.0.0 后解决
2. Device 0 被占用 → 换 device 2 后解决
3. 校准推理成功（NPU 上跑了 8 分钟） → 但编译阶段报错：
   ```
   Node[PartitionedCall_/.../Transpose_...] has no const input
   ```

**根因：** ATC 内部 AMCT 校准后重建量化图时，将 Transpose 包入 PartitionedCall 子图，导致 Transpose 的权重输入不再被识别为常量。amct_onnx 通过 TransposeFoldPass 预处理避免了此问题，但 ATC 内部不做此 pass。

**可能的解决方向：**
- 用 amct_onnx 的 TransposeFoldPass 预处理 ONNX，再喂给 ATC `--compression_optimize_conf`
- 但 TransposeFoldPass 依赖 Configuration 初始化，无法独立运行
- 如果华为在后续 CANN 版本中修复 ATC 内部 AMCT 的 TransposeFold 逻辑，此路径即可走通

### 路径 C: amct_onnx + NPU 校准 (CANNExecutionProvider) ❌

**做法：** 用 onnxruntime-cann 在 NPU 上跑校准推理。

**失败原因：** CANNExecutionProvider 使用整图编译模式（aclgrphBuildModel），AMCT 校准算子（ifmr/search_n）无 NPU kernel，图编译失败。onnxruntime 不支持对 CANN 子图做算子级 fallback。

### 路径 D: --enable_compress_weight ❌

**做法：** ATC 内置权重压缩。

**结果：** OM 大小不变（3.4G = FP16），无实际效果。

### 路径 E: amct_pytorch → torch.onnx.export ❌

**做法：** 用 amct_pytorch 校准 + convert，然后导出 ONNX。

**失败原因：** `amct_pytorch.convert()` 插入 NPU 专有算子 `npu::npu_quantize`，torch.onnx.export 不认识。

### 路径 F: Gemm 替换 MatMul+Transpose ❌

**做法：** 将 ONNX 中 `MatMul(x, Transpose(weight))` 替换为 `Gemm(x, weight, transB=1)`。

**失败原因：** Gemm 只支持 2D 输入，模型激活是 3D `[batch, seq, hidden]`。

---

## 当前可行的路径

### amct_onnx + CPU 校准 → ATC 编译

```
FP16 ONNX (310b change_node, 无自定义算子)
    ↓ amct_onnx (TransposeFoldPass + 权重校准 + 激活校准)
    ↓ onnxruntime CPU 推理 (20-30min/sample, 一次性)
    ↓ save_model → deploy_model.onnx (INT8 权重 + scale)
    ↓ ATC 编译 (CANN 9.0.0, ~2min)
    ↓ 量化 OM (NPU 推理)
```

**限制：** 校准推理在 CPU 上跑（~20-30 分钟/样本），这是一次性成本。

**使用命令：**
```bash
bash scripts/amct_onnx_ptq.sh --npu_id=3 --num_samples=8
```

---

## 待验证

1. 全零 KV cache + 多样本校准后，推理质量是否正常
2. 量化 OM 与 FP16 OM 的吞吐量对比
3. 量化 OM 的输出质量（think 是否正常、生成是否连贯）

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `export/quantize/w8a8_linear.py` | msmodelslim W8A8 权重加载器（用于 ONNX 导出） |
| `export/quantize/change_node_quant.py` | 量化图改写（QuantBatchMatmulV3，ATC 不支持） |
| `scripts/amct_onnx_calibrate.py` | AMCT ONNX 校准 Python 脚本 |
| `scripts/amct_onnx_ptq.sh` | PTQ 量化 + ATC 编译 Shell 调度脚本 |
| `export/quantize/gen_calibration_data.py` | ATC bin 校准数据生成（路径 B 用，已不可行） |
| `output/onnx_changed_310b/` | 无 RotaryMul 的标准 ONNX（用于 AMCT 校准） |
