# 量化优化探索记录 (Quantization Handoff)

## 目标

在 DeepSeek-R1-Distill-Qwen-1.5B (Ascend 910, CANN 9.0.0) 上实现 INT8 量化加速，降低 BatchMatMulV2 耗时（当前占总推理时间 75.8%）。

## 环境信息

| 项目 | 版本 |
|------|------|
| CANN | 9.0.0 |
| SoC | Ascend910_9382 (Atlas A3) |
| torch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2 |
| onnxruntime | 1.23.2 |
| AMCT | 1.1.0 (源码安装自 /mnt/host-model/cxj/amct) |
| conda env | qwen_ascend_cann900 |

## 硬件量化能力确认

Ascend910_9382 原生支持 `QuantBatchMatmulV3` 算子：
```
QuantBatchMatmulV3(x1=INT8, x2=INT8, scale=float32, pertoken_scale=float32) → output=FP16
```
定义位置：`/usr/local/Ascend/cann-9.0.0/opp/built-in/op_graph/inc/matrix_calculation_ops.h`

torch_npu 也暴露了对应接口：`torch_npu.npu_quant_matmul`, `torch_npu.npu_weight_quant_batchmatmul` 等。

---

## 方案 A：atc `--enable_compress_weight`

**状态：编译成功 ✅，但非量化**

**做法：** 在 atc 编译命令中加 `--enable_compress_weight=true`

**结果：**
- OM 编译成功，文件大小不变（3.4G）
- 这不是 INT8 量化，而是硬件级内存传输压缩（DDR→AICore 传输时压缩/解压）
- 计算仍为 FP16 MatMul
- 运行时带宽可能有收益，需 benchmark 验证

**产物：** `output/om_v4_noexpand_compress/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om`

**代码改动：** `export/onnx2om.py` 新增 `--enable_compress_weight` 参数

---

## 方案 B-1：onnxruntime 量化 + DequantizeLinear

**状态：编译失败 ❌**

**做法：** 用自定义脚本 (`export/quantize/quantize_weights.py`) 将 ONNX 模型中 196 个 Linear 层的 FP16 权重替换为 INT8 + per-channel scale，插入 `DequantizeLinear` 节点。

**结果：**
- 量化后 ONNX 模型从 3454MB 降到 2208MB（-36%），196 个 DequantizeLinear 节点正确生成
- atc 编译报错：`No operator plugin is registered for Op: DequantizeLinear`
- **原因：** atc 的 ONNX parser 不支持 `DequantizeLinear` 算子（这是标准 ONNX 量化算子，但昇腾不认）

---

## 方案 B-2：atc `--compression_optimize_conf calibration`

**状态：编译失败 ❌**

**做法：**
1. 生成校准数据 bin 文件 (`export/quantize/gen_calibration_data.py`)
2. 生成 `compression_optimize.cfg` 配置文件
3. atc 编译时加 `--compression_optimize_conf=compression_optimize.cfg`

**结果：**
- atc 报错：`Param:handle is nullptr, check invalid[FUNC:CallAmctInterface][FILE:main_impl.cc][LINE:1335]`
- **原因：** CANN 9.0.0 不包含 `libamctacl.so`（C++ AMCT 库）。从 CANN 9.0 开始，华为将 AMCT 从 toolkit 内嵌的 C++ 库改为独立开源 Python 工具。atc 中 `--compression_optimize_conf` 参数虽然还存在，但底层 `CallAmctInterface` 接口已无实现。
- 旧版 ascend-toolkit 8.3.RC1 中有 `libamctacl.so`（路径 `/usr/local/Ascend/ascend-toolkit/8.3.RC1/aarch64-linux/lib64/libamctacl.so`），但与 CANN 9.0.0 的 atc 二进制 ABI 不兼容（混用导致 segfault）。

**验证：**
```bash
find /usr/local/Ascend/cann-9.0.0 -name "libamct*"    # 空，无结果
find /usr/local/Ascend/cann-8.5.0 -name "libamct*"    # 空，无结果
find /usr/local/Ascend/cann-9.1.0 -name "libamct*"    # 空，无结果
find /usr/local/Ascend/ascend-toolkit/8.3.RC1 -name "libamct*"  # 有！但版本不兼容
```

---

## 方案 C：PyTorch 侧 W8X8 量化 → ONNX → change_node → atc

**状态：编译成功但量化无效 ⚠️**

**做法：**
1. 实现 `export/quantize/quantize_linear.py`：`W8X8Linear` 模块替换 `nn.Linear`
2. `export/export_onnx.py` 新增 `--quantize W8X8` 参数
3. 导出 ONNX 后经 `change_node_v4_noexpand.py` 做 RoPE 融合
4. atc 编译

**导出的 ONNX 图模式：**
```
weight_int8 (INT8 initializer) → Cast(to FP16) → Mul(w_scale) → Transpose → MatMul(FP16 activation, ...)
```

**结果：**
- ONNX 导出成功（6824 节点，196 层量化）
- change_node 成功（RoPE 融合 → 6488 节点）
- atc 编译成功，OM 文件 3.4G（与 FP16 baseline 相同）
- **无效原因：** atc 默认开启常量折叠（`--oo_constant_folding=true`），将 `INT8 → Cast(FP16) → Mul(scale)` 这条全常量链预计算为 FP16 张量。最终 OM 存的还是 FP16 权重，计算也还是 FP16 MatMul。

**尝试关闭常量折叠：**
- 加 `--oo_constant_folding=false` 后编译失败
- 错误：`RunGraphFusion unsuccessfully` — 图融合引擎不认识 INT8 常量→Cast→Mul→MatMul 模式

**尝试手动插入 QuantBatchMatmulV3：**
- 在 `change_node` 中将量化 pattern 替换为 `AscendQuant + QuantBatchMatmulV3`（`export/quantize/change_node_quant.py`）
- 编译失败：`No operator plugin is registered for Op: QuantBatchMatmulV3, optype: ai.onnx::14::QuantBatchMatmulV3`
- **原因：** `QuantBatchMatmulV3` 是 GE（Graph Engine）内部算子，不注册在 ONNX parser 中，不能通过 ONNX 图直接使用

**关键发现：** `change_node` 中已有的 `Cast(to=INT8) → AscendQuant` hook 检测的是激活量化方向（FP16→INT8），而我们图中的 Cast 是权重反量化方向（INT8→FP16），所以 hook 从未触发。

---

## 方案 D：AMCT PTQ + Deploy

**状态：量化成功 ✅，但产出格式不直接适用于 ONNX→OM**

**做法：**
1. 安装 AMCT：`pip install -e /mnt/host-model/cxj/amct`
2. 编写 Qwen2 适配器（`/mnt/host-model/cxj/amct/amct_pytorch/common/models/llm/qwen/qwen2/`）
3. 提取校准数据：`python -m amct_pytorch.extract_ptq_data`
4. 运行 PTQ：`python -m amct_pytorch.ptq`
5. 导出部署模型：`python -m amct_pytorch.deploy`

**结果：**
- 全流程跑通，28 层 MLP 成功量化
- 产出 `output/amct_deploy/`：量化后的 safetensors 格式模型（2.3G，原始 3.0G）
- 每层权重格式：`weight=INT8 [out, in]` + `weight_scale=float32 [out, 1]`
- config.json 中含 `quantization_config`（compressed_tensors 格式，activation 为 per-token dynamic INT8）

**局限：** AMCT deploy 产出是 HuggingFace safetensors 格式，设计给 vLLM/MindIE 等推理框架使用。不能直接走 ONNX→OM 路径。

**依赖修复记录：**
- `torch` 被 compressed_tensors 升级到 2.13.0 → 手动降回 `torch==2.7.1+cpu`
- `torchao` 版本过高 → 降为 `torchao<0.8`
- `transformers` 需要 5.12.1（AMCT 要求）
- `zstandard` 缺失 → 手动安装

---

## 尚未尝试的可行路径

### 路径 1：torch_npu 量化算子 → torchair 图编译 → OM

torch_npu 提供了原生量化 MatMul 接口：
- `torch_npu.npu_quant_matmul(x1_int8, x2_int8, scale)`
- `torch_npu.npu_weight_quant_batchmatmul(x_fp16, weight_int8, antiquant_scale)`
- `torch_npu.npu_dynamic_quant(x_fp16)` → (x_int8, scale)

思路：
1. 用 AMCT 量化后的 INT8 权重 + scale
2. 修改 modeling 文件，在 Linear forward 中调用 `npu_weight_quant_batchmatmul`
3. 用 torchair（torch.compile with npu backend）编译为静态图
4. 或通过 torch_npu 的 `torch.onnx.export` + 自定义算子映射导出 ONNX

**优势：** 绕过 ONNX parser 限制，直接使用 NPU 原生量化算子
**代价：** 需要改写 modeling 文件 + 可能需要 torchair 图编译替代 atc

### 路径 2：使用 CANN 8.3 的 atc + libamctacl

如果模型可以在 CANN 8.3 的运行时环境下执行：
1. 使用 ascend-toolkit 8.3.RC1 的 atc（自带 libamctacl.so）
2. 执行 `--compression_optimize_conf calibration`
3. 产出 OM 模型

**风险：** 编译出的 OM 可能与 CANN 9.0.0 运行时不兼容

### 路径 3：MindIE 推理框架

AMCT deploy 的产出（compressed_tensors 格式）可以直接被 MindIE 加载推理，无需经过 ONNX→OM。这是 CANN 9.0+ 官方推荐的 LLM 量化推理路径。

---

## 文件清单

**新增/修改的文件（在 qwen-ascend-llm 仓库中）：**

| 文件 | 用途 |
|------|------|
| `export/onnx2om.py` | 新增 `--enable_compress_weight`, `--compression_optimize_conf`, `--disable_constant_folding` 参数 |
| `export/export_onnx.py` | 新增 `--quantize` 参数 (none/W8X8/W8A16) |
| `export/quantize/quantize_linear.py` | PyTorch W8X8Linear/W8A16Linear 模块 |
| `export/quantize/quantize_weights.py` | ONNX 级权重量化脚本 |
| `export/quantize/gen_calibration_data.py` | atc calibration 校准数据生成 |
| `export/quantize/change_node_quant.py` | 尝试插入 QuantBatchMatmulV3（失败） |
| `scripts/onnx2om_compress_weight.sh` | 方案 A 编译脚本 |
| `scripts/quantize_w8a16.sh` | 方案 B-1 脚本 |
| `scripts/quantize_calibration_ptq.sh` | 方案 B-2 脚本 |
| `scripts/quantize_pytorch_w8.sh` | 方案 C 脚本 |

**新增文件（在 amct 仓库中）：**

| 文件 | 用途 |
|------|------|
| `amct_pytorch/common/models/llm/qwen/qwen2/__init__.py` | Qwen2 注册 |
| `amct_pytorch/common/models/llm/qwen/qwen2/qwen2.py` | Qwen2 模型适配器 |
| `amct_pytorch/common/models/llm/qwen/qwen2/quant_module.py` | Qwen2 量化模块（无 q_norm/k_norm） |
| `amct_pytorch/common/models/llm/__init__.py` | 注册 qwen2 |

---

## 核心结论

1. **CANN 9.0.0 的 ONNX→OM 路径不支持用户自定义 INT8 量化图。** atc 的 ONNX parser 不注册 QuantBatchMatmulV3/DequantizeLinear 等量化算子，`--compression_optimize_conf calibration` 因缺少 libamctacl.so 而失效。

2. **CANN 9.0+ 的 LLM 量化推理架构已变更：** 量化在 PyTorch 侧通过 AMCT 完成，部署通过 vLLM/MindIE 等框架的 compressed_tensors 支持实现，不再走 ONNX→atc→OM 路径。

3. **如果坚持 ONNX→OM 路径实现 INT8，** 最可行的方式是路径 1：在 modeling 中使用 `torch_npu.npu_weight_quant_batchmatmul` 等原生算子，让 atc 通过自定义算子机制识别。
