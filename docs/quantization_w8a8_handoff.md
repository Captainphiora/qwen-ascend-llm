# W8A8 量化工作交接

> 日期: 2026-09-08
> 分支: `quantize/w8a8`
> 基础版本: v5_gate_up_fuse

---

## 当前状态

### 已完成

1. **amct_onnx 量化全流程跑通**（校准 → deploy ONNX → change_node → ATC → OM → 推理）
2. **310B1 OM 已交付**：`opt_models/v5_quant_w8a8_310b/` 下 prefill=1/2/8 三个版本
3. **910 profiling 对比完成**：`work_logs/v5_fp16_vs_w8a8_profiling_analysis.md`
4. **310B1 profiling 对比完成**：`work_logs/v5_fp16_vs_w8a8_comparison.md`（你做的）
5. **量化流程指南**：`scripts/quantization_guide.md`
6. **测试流程文档**：`scripts/test_fp16_vs_w8a8.md`

### 关键结论

| 平台 | FP16 → W8A8 加速 | 原因 |
|------|------------------|------|
| 910 | 持平 (6.88→6.94ms) | 搬运量仅降 6.7%，带宽利用率降 8pp |
| 310B1 | **+9.4%** (188.6→172.4ms) | 权重搬运省 21%，INT8 MatMul 加速 35-80% |

310B1 比 910 收益更明显，因为 310B1 带宽更低（51.2 vs 1600 GB/s），权重搬运占比更高。

### 待解决

- 910 的 v4 完整量化版 ATC 编译失败（change_node 外部数据路径问题，非量化问题）
- INT4 量化未做

---

## PyTorch → ONNX → OM 完整量化工作流

### 前置条件

```bash
source /mnt/host-model/cxj/npu_workflow_demo/npu_env.sh qwen_ascend_cann900
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/cann-9.0.0/lib64/plugin/opskernel:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PATH=/mnt/host-model/cxj/npu-tools/profiling:/usr/local/Ascend/cann-9.0.0/bin:/usr/local/Ascend/cann-9.0.0/tools/profiler/bin:$PATH
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp
export PYTHONPATH=/mnt/host-model/cxj/qwen-ascend-llm:/usr/local/Ascend/cann-9.0.0/python/site-packages:/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /mnt/host-model/cxj/qwen-ascend-llm
```

### Step 1: ONNX 导出（如果还没有 raw ONNX）

v5 的 raw ONNX 已有：`opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx`

如需重新导出：
```bash
cd export && cp modeling_qwen2_v5_gate_up_fuse.py modeling_qwen2.py
python export_onnx.py \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --onnx_model_path <output.onnx> \
  --kv_cache_length 4096 --kv_cache_layout BSHD \
  --device_str npu --dtype float16
```

### Step 2: amct_onnx 校准

```bash
# 创建输出目录
mkdir -p opt_models/<VERSION>/amct_output

# 校准（只 skip lm_head，其余交给 amct 自动判断）
# 50 条 boolq 完整 prompt prefill，用推理引擎 KVCacheManager 构造输入
python3 scripts/amct_onnx_calibrate.py \
  --model_path opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --output_dir opt_models/<VERSION>/amct_output \
  --num_samples 50 \
  --kv_cache_length 4096 \
  --cpu_threads 64
```

注意：`scripts/amct_onnx_calibrate.py` 内部自动：
- 构建 skip_layers（默认只跳 lm_head）
- create_quant_config + quantize_model + 校准推理 + save_model
- 拷贝缺失的外部数据文件

产出：`model_deploy_deploy_model.onnx`（部署用）+ `model_deploy_fake_quant_model.onnx`（精度验证用）

### Step 3: 精度验证

用 `scripts/verify_fakequant.py` 对比 FP16 和 fake_quant ONNX 的精度。
脚本使用推理引擎的 KVCacheManager 构造输入（完整 prompt prefill + decode loop 生成），
和实际推理行为一致。

```bash
# 验证单个版本
python3 scripts/verify_fakequant.py \
  --fp16_onnx opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
  --fakequant_onnx opt_models/<VERSION>/amct_output/model_deploy_fake_quant_model.onnx \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --output opt_models/<VERSION>/verify_result.txt

# 同时对比多个版本
python3 scripts/verify_fakequant.py \
  --fp16_onnx opt_models/v5_gate_up_fuse/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx \
  --fakequant_onnx \
    v3_skip85=opt_models/v5_quant_w8a8/amct_output_v3/model_deploy_fake_quant_model.onnx \
    v4_skip_lm_head=opt_models/v5_quant_w8a8/amct_output_v4/model_deploy_fake_quant_model.onnx \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B
```

输出内容：
- Prefill logits 的 cosine similarity、max abs diff
- Top-1 / Top-5 匹配度
- 完整生成文本（通过引擎 decode loop）
- 自动判定：PASS (>0.8) / WARN (0.6-0.8) / FAIL (<0.6)

已验证结果（v5 W8A8）：

| 版本 | skip 策略 | Cosine | Top-1 | Top-5 | 判定 |
|------|----------|--------|-------|-------|------|
| v3 | skip 85 层 (QK/SV + fused MLP + lm_head) | 0.843 | Y (嗯) | 5/5 | PASS |
| v4 | 只 skip lm_head | 0.695 | N (Okay) | 3/5 | WARN |

### Step 4: change_node

```bash
# 910（RoPE 融合）
python3 export/change_node_v5_gate_up_fuse.py \
  --input opt_models/<VERSION>/amct_output/model_deploy_deploy_model.onnx \
  --output opt_models/<VERSION>/deploy_changed_910.onnx

# 310B1（Trilu 修复，不做 RoPE 融合）
python3 export/change_node_v4_noexpand_310b.py \
  --input_model_path opt_models/<VERSION>/amct_output/model_deploy_deploy_model.onnx \
  --output_model_path opt_models/<VERSION>/deploy_changed_310b.onnx
```

### Step 5: ATC 编译

```bash
python3 export/onnx2om.py \
  --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --onnx_model_path opt_models/<VERSION>/deploy_changed_<PLATFORM>.onnx \
  --om_model_path opt_models/<VERSION>/<MODEL_NAME> \
  --kv_cache_length 4096 \
  --max_prefill_length 1 \
  --kv_cache_layout BSHD \
  --precision_mode origin \
  --soc_version <Ascend910_9382 | Ascend310B1> \
  --cpu_thread 16
```

关键参数：
- `--precision_mode origin`：AMCT 量化模型必须用 origin
- `--max_prefill_length 1`：静态 OM 2.9GB；≥2 动态 OM 5.3GB（+2.4GB 固定开销）

### Step 6: 推理验证 + Benchmark

```bash
# Benchmark
python3 benchmarks/benchmark.py \
  --om_model_path <om_path> \
  --device_id $DEVICE_ID \
  --max_prefill_length 1 \
  --rounds 5 --warmup 2 --max_new_tokens 50 \
  --label "<label>"
```

### Step 7: Profiling

```bash
# 采集（必须加 --aic-metrics=PipeUtilization 才有 mac/mte2 ratio）
npu-profile.sh \
  "python3 benchmarks/profile_decode.py \
    --om_model_path <om_path> \
    --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device_id $DEVICE_ID --max_new_tokens 30" \
  --label <label> \
  --output-dir profiling/<label> \
  --msprof-args "--aic-metrics=PipeUtilization"

# 搬运量 / 带宽利用率 / 差距分析
python3 scripts/calc_theoretical_throughput.py \
  --config_json /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B/config.json \
  --profiling_csv profiling/<label>/PROF_*/mindstudio_profiler_output/op_summary_*.csv \
  --peak_bw <1600|51.2> \
  --peak_flops <376|10> \
  --weight_dtype_size <1|2>   # INT8=1, FP16=2
```

---

## 文件索引

| 文件 | 说明 |
|------|------|
| `scripts/quantization_guide.md` | 量化完整流程指南（含 INT4 适配要点和踩坑记录） |
| `scripts/test_fp16_vs_w8a8.md` | FP16 vs W8A8 手动测试流程 |
| `scripts/amct_onnx_calibrate.py` | amct_onnx 校准脚本（已修复外部数据拷贝） |
| `scripts/calc_theoretical_throughput.py` | 理论峰值 / 搬运量 / 利用率分析（支持 INT8 权重） |
| `opt_logs/QUANTIZATION_W8A8_LOG.md` | 完整工作日志 |
| `work_logs/v5_fp16_vs_w8a8_profiling_analysis.md` | 910 profiling 对比分析 |
| `work_logs/v5_fp16_vs_w8a8_comparison.md` | 310B1 profiling 对比分析 |
| `/mnt/host-model/cxj/npu-tools/profiling/npu-profile.sh` | 通用 profiling 工具 |
| `/mnt/host-model/cxj/npu-tools/profiling/parse_profiling.py` | 通用 profiling 分析脚本 |

## 量化产物索引

| 目录 | 内容 |
|------|------|
| `opt_models/v5_quant_w8a8/amct_output_v3/` | v3 校准产出（skip 85 层，cosine 0.84） |
| `opt_models/v5_quant_w8a8/amct_output_v4/` | v4 校准产出（只 skip lm_head，cosine 0.69） |
| `opt_models/v5_quant_w8a8/v3_*.om` | v3 910 OM（2.9GB） |
| `opt_models/v5_quant_w8a8_310b/` | 310B1 OM（prefill=1/2/8） |

---

## 踩坑清单

### #1 校准输入构造错误导致精度崩溃（最关键的教训）

**现象**：amct_onnx 校准 → fake_quant ONNX 推理 → 输出全乱码，cosine < 0.2

**错误做法**（v1/v2 校准）：
```python
# 错误：每条 prompt 只取最后 1 个 token，KV cache 全零
input_ids = np.array([[tokens[prompt_len - 1]]], dtype=np.int64)  # shape [1, 1]
past_key_values = np.zeros((1, 4096, 112, 128), dtype=np.float16)  # 全零
attention_mask = np.ones((1, 1 + 4096), dtype=np.int64)
```

这样做的后果：模型看到的是"空上下文 + 单个随机 token"，所有层的激活值都接近零或默认值。
AMCT 基于这些不真实的激活分布计算出的 scale/offset 完全不适用于真实推理场景。

**正确做法**（v3/v4 校准）：
```python
# 正确：完整 prompt 作为 input_ids，用推理引擎的 KVCacheManager 构造其余输入
tokens = tokenizer.encode(prompt)
input_ids = np.array([tokens[:seq_len]], dtype=np.int64)  # shape [1, seq_len] 完整 prompt
kv_mgr = create_kv_cache(inf_config)  # 项目推理引擎的 KV cache 管理器
cache, mask, pos_ids = kv_mgr.get_inputs(seq_len)  # 自动生成正确的 mask/position_ids/cache
```

**为什么必须用推理引擎的 KVCacheManager**：
- `attention_mask` 的格式和尺寸由引擎决定（不是简单的全 1 矩阵）
- `position_ids` 的起始位置和长度由引擎决定
- `past_key_values` 的 shape 由 `InferenceConfig` 的 `kv_cache_layout` 决定（BSHD vs BHSD）
- 手动构造这些输入很容易出错，而且和实际推理时的输入不一致

**发现过程**：
1. 最初用手写 numpy 输入做校准和验证，cosine 崩溃到 -0.09（v1）和 0.16（v2）
2. 怀疑是量化配置问题（skip_layers、channel_wise 等），调了多次都不行
3. 用 msmodelslim 做 PyTorch 侧量化也崩溃（cosine 0.16），排除了 amct_onnx 的问题
4. 发现官方量化权重反量化后同样乱码，认识到量化权重不能直接用标准 PyTorch 推理
5. 最终回到 amct_onnx 路径，用项目推理引擎的 OnnxSession 做验证，发现 FP16 ONNX 通过引擎推理输出正常——**问题不在验证方式，而在校准输入的构造方式**
6. 改用 KVCacheManager + 完整 prompt prefill 校准后，cosine 恢复到 0.84

### 其他踩坑

| # | 问题 | 根因 | 解决 |
|---|------|------|------|
| 2 | modified ONNX 加载失败 | amct 未拷贝 RoPE 常量外部数据文件 | 校准脚本自动拷贝 |
| 3 | msmodelslim 量化后推理乱码 | ascendV1 格式给 MindIE 用，不能走 nn.Linear | 改用 amct_onnx |
| 4 | 310B1 不能用 change_node_v5 | 310B1 不支持 NPURotaryPositionEmbedding | 用 change_node_v4_noexpand_310b |
| 5 | OM 从 2.9G 跳到 5.3G | max_prefill≥2 开启 dynamic_dims 固定开销 | 与档位数和 kv_len 无关 |
| 6 | config.json torch_dtype=float16 | 原始权重是 BF16，CPU 上 BF16→FP16 溢出 | NPU 上不影响；校准在 CPU onnxruntime 上跑不受影响 |
| 7 | W8A8 在 910 上没加速 | 权重搬运省 42% 但中间结果搬运增 16%，带宽利用率降 8pp | 1.5B 小模型 batch=1 是访存瓶颈，中间结果主导 |
