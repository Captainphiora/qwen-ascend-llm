# W8A8 量化校准方法论与实验记录

> 基础版本: v5_gate_up_fuse
> 模型: DeepSeek-R1-Distill-Qwen-1.5B (28层, hidden_size=1536, kv_heads=2)
> 工具: amct_onnx (CANN 9.0.0)
> 平台: Atlas 800T A3 (Ascend 910)

---

## 一、AMCT 量化流程概述

amct_onnx 的 W8A8 PTQ（Post-Training Quantization）分四步：

```
create_quant_config → quantize_model → 校准推理(N次) → save_model
        ↓                   ↓                ↓              ↓
  quant_config.json   modified.onnx    record.txt     deploy.onnx + fake_quant.onnx
```

1. **create_quant_config**: 扫描 ONNX 图，为每个 MatMul 生成量化配置（skip/量化、IFMR参数、ARQ参数）
2. **quantize_model**: 在每个被量化的 MatMul 前后插入统计节点，产出 modified.onnx
3. **校准推理**: 用校准数据跑 modified.onnx，统计节点收集每层激活的 min/max 分布，写入 record.txt
4. **save_model**: 根据 record.txt 中的 scale 信息，产出：
   - `deploy_model.onnx`: 真正的 INT8 量化模型（给 ATC 编译用）
   - `fake_quant_model.onnx`: 仿真量化模型（FP16 精度模拟 INT8 行为，用于精度验证）

## 二、.cfg 配置文件详解

当使用 `config_defination` 参数时，通过 `.cfg` 文件（protobuf text 格式）控制量化行为。
使用 `config_defination` 时，`skip_layers` 必须写在 .cfg 文件内，不能通过 API 参数传入。

### 完整示例（当前最优配置 v7_skip88）

```protobuf
# 校准批次数，需与实际校准样本数一致
batch_num : 50
# 激活量化是否带 offset（非对称量化），推荐 true
activation_offset : true

# === skip_layers: 不量化的层 ===
# 格式: skip_layers : "ONNX节点名"
# 可用 Netron 查看 ONNX 图中的节点名

# 1. lm_head: 输出层，精度敏感，必须跳过
skip_layers : "/lm_head/MatMul"

# 2. Attention QK^T (28层): 两端都是激活(Q*K)，无固定权重
# 3. Attention Score*V (28层): 两端都是激活(attn_score*V)，无固定权重
# 4. MLP fused gate_up (28层): 两端都是激活(gate_output*up_output)
# 以上 84 层是 activation×activation 的 MatMul，不适合 W8A8
skip_layers : "/model/layers.0/self_attn/MatMul"
skip_layers : "/model/layers.0/self_attn/MatMul_1"
skip_layers : "/model/layers.0/mlp/MatMul"
# ... (每层3个，共28层 = 84个)

# 5. outlier down_proj (3层): 激活存在极端离群值，量化误差过大
#    Layer 2: abs_max=3085, p99.99=11.1, 离群比=278x
#    Layer 26: abs_max=1225, p99.99=106, 离群比=11.5x
#    Layer 27: abs_max=474, p99.99=232, 离群比=2.0x
skip_layers : "/model/layers.2/mlp/down_proj/MatMul"
skip_layers : "/model/layers.26/mlp/down_proj/MatMul"
skip_layers : "/model/layers.27/mlp/down_proj/MatMul"

# 总计 skip 88 层，量化 137 层（均为有固定权重的投影层）

# === 量化算法配置 ===
common_config : {
    # ARQ: 权重量化算法 (Adaptive Range Quantization)
    arq_quantize : {
        # channel_wise: 是否按 output channel 分别量化
        # amct_onnx 对 MatMul 不支持 true，只对 Gemm/Conv 有效
        # 如需 per-channel，需先将 ONNX 中的 MatMul 转为 Gemm
        channel_wise : false
    }
    # IFMR: 激活量化算法 (Intelligent Flexible Min-max Range)
    ifmr_quantize : {
        # 在 [percentile * search_range_start, percentile * search_range_end] 范围内
        # 以 search_step 为步长搜索最优 clipping threshold
        search_range_start : 0.7
        search_range_end : 1.3
        search_step : 0.01
        # max/min_percentile: 用数据的第几分位数作为搜索基准
        # 0.999999 (6个9): 接近 abs_max，被极端 outlier 主导
        # 0.9999 (4个9): 截断 outlier，但对本模型截断过激导致精度崩溃
        # 建议: 保持 0.999999，通过 skip_layers 跳过 outlier 严重的层
        max_percentile : 0.999999
        min_percentile : 0.999999
    }
}
```

### 参数选择依据

| 参数 | 当前值 | 为什么 |
|------|--------|--------|
| batch_num | 50 | boolq.jsonl 共 50 条，全部用于校准 |
| activation_offset | true | 非对称量化，能更好覆盖非零中心的激活分布 |
| channel_wise | false | amct_onnx 限制：MatMul 不支持 per-channel |
| max_percentile | 0.999999 | 保守策略：不截断 outlier，靠 skip_layers 处理问题层 |
| search_range | [0.7, 1.3] | IFMR 默认值，在基准值的 70%~130% 范围搜索最优 scale |

### skip_layers 分类逻辑

v5 ONNX 共 225 个 MatMul，分四类：

| 类型 | 数量 | 特征 | 是否量化 | 原因 |
|------|------|------|----------|------|
| 权重投影 (q/k/v/o/gate/up/down_proj) | 196 | 一端是固定权重 W | 大部分是 | W8A8 的标准目标 |
| Attention QK^T / Score*V | 56 | 两端都是激活 | 否 | 无权重，量化两端激活误差大 |
| MLP fused gate_up | 28 | 两端都是激活 | 否 | SiLU(gate)*up，无权重 |
| lm_head | 1 | 输出层 | 否 | 精度最敏感 |
| outlier down_proj (L2/26/27) | 3 | 输入有极端离群值 | 否 | abs_max/p99.99 > 10x |

最终: skip 88 层, 量化 137 层。

## 三、校准数据与输入构造

### 校准数据

使用 boolq.jsonl（BoolQ 阅读理解数据集），每条格式：
```json
{"id": 0, "inputs_pretokenized": "Ghost in the Shell -- ...\\nQuestion: is ghost in the shell based on the anime?\\nAnswer:\\n"}
```

50 条样本，token 长度分布约 69~412，覆盖不同长度的输入。

### 输入构造（完整 prefill）

以第一条数据（315 tokens）为例：

```
原始文本 → apply_chat_template → tokenize → 构造4个输入tensor → 喂给模型
```

```python
# 1. 包装成对话格式
messages = [{'role': 'user', 'content': prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# 结果: "<｜begin▁of▁sentence｜><｜User｜>Ghost in the Shell -- ...<｜Assistant｜>"

# 2. Tokenize
tokens = tokenizer.encode(text)  # → 315 个 token ID

# 3. 构造模型输入
input_ids      = [tokens]                    # shape=(1, 315) 完整 prompt
attention_mask = [1,1,...,1, 0]              # shape=(1, 316) 315个1 + 空KV的1个0
position_ids   = [0, 1, 2, ..., 314]        # shape=(1, 315)
past_key_values = zeros(1, 1, 112, 128)     # shape=(1, 1, kv_dim, head_dim) 空KV cache
```

### 为什么用完整 prefill 而非单 token

| 方式 | 做法 | 问题 |
|------|------|------|
| 旧方式 | input_ids=[最后1个token], KV=全零(4096长) | Q与全零K点积≈0，attention均匀分布，激活分布不真实 |
| 新方式 | input_ids=[全部315个token], KV=空(长度1) | 模型计算真实 causal attention，激活分布正确 |

完整 prefill 和推理引擎的逐 token decode 在数学上等价：causal mask 保证 token i 只 attend 到 position 0..i-1，
无论是一次性计算还是逐步累积 KV cache，每个位置的 attention 结果相同。

### 模型在校准中做了什么

315 个 token 经过 28 层 transformer：
```
input_ids → embed_tokens → [Layer 0 → Layer 1 → ... → Layer 27] → lm_head → logits
```
每层内部：
```
hidden_states → [RMSNorm → q/k/v_proj → RoPE → Attention → o_proj] → residual
             → [RMSNorm → gate_proj → SiLU → *up_proj → down_proj] → residual
```

amct 统计节点挂在每个被量化的 MatMul 前后：
- 权重侧: 记录 W 矩阵的 min/max → scale_w
- 激活侧: 记录 315 个 token 的 hidden_states min/max → 累积到 scale_d

50 条样本后，IFMR 算法根据累积的分布信息，在 search_range 内搜索最优 scale。

## 四、精度验证

### 工具

`scripts/verify_fakequant.py`: 对比 FP16 raw ONNX 和 fake_quant ONNX

验证内容：
1. Prefill logits cosine similarity（整体精度指标）
2. Top-5 token 匹配度（排序稳定性）
3. 完整 decode 生成文本（端到端质量）

### 合格标准

| 指标 | 要求 |
|------|------|
| Cosine similarity | > 0.95 |
| Top-1 match | True |
| Top-5 overlap | >= 4/5 |
| 生成文本 | 语言正确，语义合理 |

## 五、实验结果

| 版本 | 校准方式 | skip 策略 | percentile | Cosine | Top-1 | Top-5 |
|------|---------|-----------|-----------|--------|-------|-------|
| v3 | 单token+零KV | 85层 | 0.999999 | 0.843 | ✓ | 5/5 |
| v4 | 单token+零KV | lm_head only | 0.999999 | 0.695 | ✗ | 3/5 |
| v5 | **完整prefill** | 85层 | 0.999999 | 0.903 | ✓ | 4/5 |
| v6 | 完整prefill | 85层 | **0.9999** | 0.080 | ✗ | 0/5 |
| **v7** | **完整prefill** | **88层** | 0.999999 | **0.965** | ✓ | **5/5** |

### 各版本改进点

- v3→v5: 修复校准输入（完整 prefill 替代单 token+零 KV）→ cosine +0.060
- v5→v7: 跳过 3 个 outlier down_proj (L2/26/27) → cosine +0.063
- v6: percentile 截断实验失败（0.9999 过于激烈）

## 六、待解决的问题

### per-channel 权重量化
amct_onnx 对 MatMul 算子不支持 `channel_wise=true`。
可能路径: ONNX 导出时将 MatMul+bias 转为 Gemm 节点。

### activation outlier 精细处理
当前方案直接跳过 outlier 层。更优方案:
- 针对问题层用 `override_layer_configs` 单独设 percentile
- SmoothQuant: 将激活离群值数学等价地转移到权重侧

### decode 阶段校准覆盖
当前只用 prefill 做校准。虽然 causal attention 保证数学等价，
但 decode 阶段的实际激活分布（单 token + 长 KV cache）可能有细微差异。
可考虑在校准中混入 decode 步骤。

## 七、产物路径

| 目录 | skip | 校准 | Cosine | 状态 |
|------|------|------|--------|------|
| amct_output_v3 | 85 | 单token | 0.843 | 旧版 baseline |
| amct_output_v4 | 1 | 单token | 0.695 | 旧版 baseline |
| amct_output_v5_cw | 85 | prefill | 0.903 | 改进校准 |
| amct_output_v6_p9999 | 85 | prefill+p9999 | 0.080 | 废弃 |
| **amct_output_v7_skip88** | **88** | **prefill** | **0.965** | **当前最优** |

## 八、复现命令

```bash
# 环境
source /mnt/host-model/cxj/npu_workflow_demo/npu_env.sh qwen_ascend_cann900
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export PYTHONPATH=/mnt/host-model/cxj/qwen-ascend-llm:$PYTHONPATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# 校准
python3 scripts/amct_onnx_calibrate.py \
  --model_path opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --output_dir opt_models/v5_quant_w8a8/amct_output_v7_skip88 \
  --num_samples 50 --kv_cache_length 4096 --cpu_threads 64 \
  --skip_layers "<88层skip list>"

# 验证
python3 scripts/verify_fakequant.py \
  --fp16_onnx opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
  --fakequant_onnx v7=opt_models/v5_quant_w8a8/amct_output_v7_skip88/model_deploy_fake_quant_model.onnx \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B
```

## 九、完整量化流程（端到端）

从 FP16 ONNX 到可部署的量化 OM，共六步：

```
Step 1          Step 2         Step 3          Step 4        Step 5         Step 6
ONNX 导出 → amct 校准 → 精度验证 → change_node → ATC 编译 → 推理验证/Benchmark
(PyTorch→ONNX) (FP16→W8A8)  (fake_quant)   (RoPE融合)    (ONNX→OM)     (OM推理)
```

### Step 1: ONNX 导出

将 PyTorch 模型导出为 FP16 ONNX。v5 的 raw ONNX 已有，无需重新导出。

```bash
cd export && cp modeling_qwen2_v5_gate_up_fuse.py modeling_qwen2.py
python3 export_onnx.py \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --onnx_model_path opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
  --kv_cache_length 4096 --kv_cache_layout BSHD \
  --device_str npu --dtype float16
```

| 项目 | 说明 |
|------|------|
| 脚本 | `export/export_onnx.py` |
| 模型定义 | `export/modeling_qwen2_v5_gate_up_fuse.py` |
| 产物 | `opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx` + 外部数据文件 |

### Step 2: amct_onnx 校准

对 FP16 ONNX 做 W8A8 PTQ 校准，产出量化 ONNX。

```bash
python3 scripts/amct_onnx_calibrate.py \
  --model_path opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --output_dir opt_models/v5_quant_w8a8/amct_output_v7_skip88 \
  --num_samples 50 --kv_cache_length 4096 --cpu_threads 64 \
  --skip_layers "<88层skip list>"
```

也可以用 .cfg 文件（包含 skip_layers 和算法参数）：
```bash
python3 scripts/amct_onnx_calibrate.py \
  --model_path ... --hf_model_dir ... \
  --output_dir ... --num_samples 50 \
  --quant_cfg scripts/quant_v7_skip88.cfg
```

| 项目 | 说明 |
|------|------|
| 脚本 | `scripts/amct_onnx_calibrate.py` |
| 配置 | `scripts/quant_v3_skip85_channel_wise.cfg` (示例，含参数注释) |
| 校准数据 | `/usr/local/Ascend/atb-models/examples/convert/model_slim/boolq.jsonl` |
| 产物 | `<output_dir>/model_deploy_deploy_model.onnx` (部署用) |
| | `<output_dir>/model_deploy_fake_quant_model.onnx` (精度验证用) |
| | `<output_dir>/record.txt` (每层 scale_d/scale_w) |
| | `<output_dir>/quant_config.json` (实际使用的量化配置) |

### Step 3: 精度验证

用 fake_quant ONNX 对比 FP16 ONNX 的 logits 和生成文本。在 ATC 编译前做，避免浪费编译时间。

```bash
python3 scripts/verify_fakequant.py \
  --fp16_onnx opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
  --fakequant_onnx v7=opt_models/v5_quant_w8a8/amct_output_v7_skip88/model_deploy_fake_quant_model.onnx \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --prompt "请用一句话介绍量子计算。" \
  --max_new_tokens 80
```

| 项目 | 说明 |
|------|------|
| 脚本 | `scripts/verify_fakequant.py` |
| 验证内容 | prefill cosine + top-5 匹配 + decode 生成文本 |
| 合格标准 | cosine > 0.95, top-1 match, 生成文本语言和语义正确 |

### Step 4: change_node（RoPE 融合）

将 ONNX 中的 RoPE 算子模式替换为昇腾 NPURotaryPositionEmbedding 融合算子。
对量化后的 deploy ONNX 和 FP16 ONNX 使用同一脚本。

```bash
# 910 版本（RoPE 融合）
python3 export/change_node_v5_gate_up_fuse.py \
  --input_model_path opt_models/v5_quant_w8a8/amct_output_v7_skip88/model_deploy_deploy_model.onnx \
  --output_model_path opt_models/v5_quant_w8a8/v7_deploy_changed_910.onnx

# 310B1 版本（Trilu 修复，不做 RoPE 融合）
python3 export/change_node_v4_noexpand_310b.py \
  --input_model_path opt_models/v5_quant_w8a8/amct_output_v7_skip88/model_deploy_deploy_model.onnx \
  --output_model_path opt_models/v5_quant_w8a8/v7_deploy_changed_310b.onnx
```

| 项目 | 说明 |
|------|------|
| 脚本 (910) | `export/change_node_v5_gate_up_fuse.py` |
| 脚本 (310B1) | `export/change_node_v4_noexpand_310b.py` |
| 注意 | 两个脚本参数名不同 (`--input/--output` vs `--input_model_path/--output_model_path`) |

### Step 5: ATC 编译（ONNX → OM）

```bash
python3 export/onnx2om.py \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --onnx_model_path opt_models/v5_quant_w8a8/v7_deploy_changed_910.onnx \
  --om_model_path opt_models/v5_quant_w8a8/v7_DeepSeek-R1-Distill-Qwen-1.5B_4096_1_w8a8 \
  --kv_cache_length 4096 --max_prefill_length 1 \
  --kv_cache_layout BSHD \
  --precision_mode origin \
  --soc_version Ascend910_9382 --cpu_thread 16
```

| 项目 | 说明 |
|------|------|
| 脚本 | `export/onnx2om.py` |
| 关键参数 | `--precision_mode origin` (量化模型必须用 origin) |
| | `--max_prefill_length 1` (静态 OM ~2.9GB; >=2 动态 OM ~5.3GB) |
| | `--soc_version Ascend910_9382` (910) 或 `Ascend310B1` (310B1) |
| 产物 | `<om_model_path>.om` |

### Step 6: 推理验证 + Benchmark + Profiling

```bash
# Benchmark
python3 benchmarks/benchmark.py \
  --om_model_path <om_path> --device_id $DEVICE_ID \
  --max_prefill_length 1 --rounds 5 --warmup 2 --max_new_tokens 50 \
  --label "v7_w8a8_skip88"

# Profiling 采集 (使用 npu-profile.sh，不依赖项目结构)
npu-profile.sh \
  "python3 benchmarks/profile_decode.py \
    --om_model_path <om_path> \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device_id $DEVICE_ID --max_new_tokens 30" \
  --label v7_w8a8 --output-dir profiling/v7_w8a8 \
  --msprof-args "--aic-metrics=PipeUtilization"

# 理论吞吐分析
python3 scripts/calc_theoretical_throughput.py \
  --config_json /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B/config.json \
  --profiling_csv profiling/v7_w8a8/PROF_*/mindstudio_profiler_output/op_summary_*.csv \
  --peak_bw 1600 --peak_flops 376 --weight_dtype_size 1
```

| 项目 | 说明 |
|------|------|
| Benchmark | `benchmarks/benchmark.py` |
| Profiling 采集 | `npu-profile.sh` (通用工具) + `benchmarks/profile_decode.py` |
| | `npu-profile.sh` 位于 `/mnt/host-model/cxj/npu-tools/profiling/`，需加入 PATH |
| | 项目内 `scripts/profile.sh` 是旧版，功能相同但不兼容 CANN 9.0+ 的 Python profiler_tool |
| Profiling 分析 | `benchmarks/parse_profiling.py` |
| 理论吞吐分析 | `scripts/calc_theoretical_throughput.py` (`--weight_dtype_size 1` 表示 INT8) |

## 十、脚本索引

### 量化相关

| 脚本 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `scripts/amct_onnx_calibrate.py` | amct 校准主脚本 | FP16 ONNX + boolq 数据 | deploy/fake_quant ONNX + record.txt |
| `scripts/verify_fakequant.py` | 精度验证 (cosine + 生成) | FP16 ONNX + fake_quant ONNX | cosine/top-5/生成文本对比 |
| `scripts/quant_v3_skip85_channel_wise.cfg` | .cfg 配置示例 (含详细注释) | - | - |

### 导出与编译

| 脚本 | 作用 |
|------|------|
| `export/export_onnx.py` | PyTorch → ONNX 导出 |
| `export/modeling_qwen2_v5_gate_up_fuse.py` | v5 模型定义 (gate_up fuse) |
| `export/change_node_v5_gate_up_fuse.py` | ONNX 图改写: RoPE 融合 (910) |
| `export/change_node_v4_noexpand_310b.py` | ONNX 图改写: Trilu 修复 (310B1) |
| `export/onnx2om.py` | ATC 编译封装 (ONNX → OM) |

### 推理与性能

| 脚本 | 作用 |
|------|------|
| `benchmarks/benchmark.py` | OM 推理 benchmark (TPOT/吞吐) |
| `benchmarks/profile_decode.py` | decode profiling 采集 |
| `benchmarks/parse_profiling.py` | profiling 深度分析 (算子/利用率) |
| `scripts/profile.sh` | msprof 采集+解析 (旧版，项目内) |
| `/mnt/host-model/cxj/npu-tools/profiling/npu-profile.sh` | msprof 采集+解析 (通用版，兼容 CANN 9.0+ Python profiler_tool) |
| `scripts/calc_theoretical_throughput.py` | Roofline 理论峰值分析 |
