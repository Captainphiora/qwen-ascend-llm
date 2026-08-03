---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 24px;
  }
  h1 {
    font-size: 36px;
  }
  h2 {
    font-size: 30px;
  }
  code {
    font-size: 18px;
  }
  table {
    font-size: 20px;
  }
  pre {
    font-size: 16px;
  }
---

# DeepSeek-R1-Distill-Qwen-1.5B 在昇腾 NPU 上的部署

## 从 HuggingFace 模型到 ACL 推理的完整技术方案

---

## 目录

1. 项目概览与技术路线
2. 模型静态化改造
3. ONNX 导出与图修正
4. ATC 编译与动态 Shape 分档
5. ACL Runtime 推理流程（完整示例）
6. KV Cache 管理机制
7. Prefill 与 Decode 全流程演示
8. 性能数据

---

## 1. 技术路线

```text
HuggingFace PyTorch 模型 (动态图, 动态 shape)
        ↓ 静态化改造 (export/modeling_qwen2.py)
        ↓ torch.onnx.export (trace)
ONNX 中间表示 (静态计算图)
        ↓ 图修正 (export/change_node.py)
        ↓ ATC 离线编译 (export/onnx2om.py)
OM 模型 (NPU 专用二进制, 预编译执行方案)
        ↓ ACL Runtime 加载与执行
昇腾 NPU 推理
```

---

## 1.1 环境配置

| 项目 | 配置 |
|---|---|
| 模型 | DeepSeek-R1-Distill-Qwen-1.5B |
| NPU | Ascend 910 |
| CANN | 9.0.0 |
| Python 环境 | conda: `qwen_ascend_cann900` |
| PyTorch | 2.1.0 + torch_npu |
| Transformers | 4.37.0 |

关键路径：
- ONNX: `output/onnx2_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx`
- OM: `output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om`

---

## 2. 模型静态化改造

### 原始 HuggingFace 模型的问题

- `forward()` 有多个可选参数和 Python 控制分支
- KV Cache 是动态增长的 tuple 列表
- Attention 实现有 eager / flash / sdpa 三种分支
- 不能直接 `torch.onnx.export`

---

## 2.1 固定输入输出接口

```python
# export/modeling_qwen2.py 中的改造
# 原始: forward(input_ids, attention_mask=None, position_ids=None, 
#              past_key_values=None, use_cache=None, ...)
# 改造后: 固定4个必需输入，2个固定输出

输入: input_ids       (batch, seq_len)         - token IDs
      attention_mask  (batch, kv_len + seq_len) - padding mask
      position_ids    (batch, seq_len)         - 位置编码
      past_key_values (batch, kv_len, K, D)    - KV Cache

输出: logits          (batch, seq_len, vocab_size) - 预测分布
      out_key_values  (batch, seq_len, K, D)       - 新生成的 KV
```

其中 `K = num_layers × 2 × num_kv_heads = 28 × 2 × 2 = 112`，`D = 128`

---

## 2.2 KV Cache 张量化

```python
# 原始 HuggingFace: 动态 tuple
past_key_values = (
    (layer0_key, layer0_value),  # 每层独立，长度随生成增长
    (layer1_key, layer1_value),
    ...
)

# 改造后: 固定形状的单个 4D 张量
past_key_values.shape = (1, 4096, 112, 128)
#                        batch, kv_cache_length, layers*2*heads, head_dim
```

每层通过切片取自己的 KV：
```python
# 第 i 层取 Key 和 Value (modeling_qwen2.py:283-290)
cache_key = past_key_values[:, i*2*heads : (i*2+1)*heads]
cache_value = past_key_values[:, (i*2+1)*heads : (i*2+2)*heads]
```

---

## 2.3 强制使用 Eager Attention

```python
# 原始: 根据 config 动态选择
self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
# 可能是 FlashAttention2 或 SDPA → 不可 trace

# 改造后: 硬编码为 eager
self.self_attn = Qwen2Attention(config, layer_idx)
# eager attention = MatMul → Scale → Add(mask) → Softmax → MatMul
# 全部是标准 ONNX 算子，NPU 可识别
```

---

## 3. ONNX 导出

### 导出命令

```bash
conda activate qwen_ascend_cann900
python export/export_onnx.py \
    --device_str=npu --dtype=float16 \
    --hf_model_dir=/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --onnx_model_path=output/onnx_xxx/model.onnx \
    --kv_cache_length=4096 --simplify=false
```

### Trace 过程

`torch.onnx.export` 用示例输入跑一遍 forward，记录所有算子调用 → 生成静态 ONNX 图

---

## 3.1 ONNX 图修正 (change_node.py)

### 问题 1: Trilu 算子

PyTorch 导出的 `Trilu` 带动态 diagonal 输入参数，昇腾 ATC 要求编译期确定：

```python
# 修正: 去掉第二个输入，改为属性
if node.op_type == "Trilu":
    new_node = helper.make_node("Trilu", 
        inputs=[node.input[0]],  # 只保留数据输入
        outputs=node.output, upper=0)  # upper=0 固化为属性
```

### 问题 2: Cast to INT8

```python
if node.op_type == "Cast" and to_attribute.i == TensorProto.INT8:
    new_node = helper.make_node("AscendQuant", ...)  # 替换为昇腾专用量化算子
```

---

## 4. ATC 编译

### 编译命令 (export/onnx2om.py 生成)

```bash
atc --framework=5 \
    --model="output/onnx2_.../model.onnx" \
    --output="output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1" \
    --soc_version=Ascend910_9382 \
    --precision_mode_v2=mixed_float16 \
    --modify_mixlist=ops_info.json \
    --input_format=ND \
    --input_shape="input_ids:1,1;attention_mask:1,4097;position_ids:1,1;past_key_values:1,4096,112,128"
```

---

## 4.1 动态 Shape 分档

当 `max_prefill_length > 1` 时，注册多组合法 shape：

```bash
--dynamic_dims "
  1,1, 1,4097, 1,1, 1,4096,112,128;    # seq=1, kv=全长
  1,2, 1,4098, 1,2, 1,4096,112,128;    # seq=2, kv=全长
  1,4, 1,4100, 1,4, 1,4096,112,128;    # seq=4, kv=全长
  1,1, 1,2049, 1,1, 1,2048,112,128;    # seq=1, kv=半长
  1,2, 1,2050, 1,2, 1,2048,112,128;    # seq=2, kv=半长
  1,4, 1,2052, 1,4, 1,2048,112,128;    # seq=4, kv=半长
"
```

每个分号 = 一组预编译的执行方案（Tiling + 内存布局 + 指令流）

---

## 4.2 ATC 做了什么

```text
ONNX (What to compute)
    ↓ 图解析
    ↓ 算子融合 (MatMul+Add → FusedOp, 减少 HBM 读写)
    ↓ 常量折叠 (编译期计算确定值)
    ↓ 算子映射 (ONNX op → 昇腾 AI Core kernel)
    ↓ 内存规划 (每个 tensor 的地址, buffer 复用)
    ↓ 指令生成 (AI Core 三级流水: MTE→CUBE→VEC)
OM (How to execute on this chip)
```

OM = 权重 + 每个 shape 档位的完整执行方案

---

## 5. ACL Runtime 推理流程

### 初始化 (engine.py)

```python
acl.init()                           # 初始化 ACL
acl.rt.set_device(device_id)         # 选择 NPU 设备
context = acl.rt.create_context(device_id)  # 创建上下文

# 加载 OM 模型到 Device 内存
model_buffer = acl.rt.malloc_host(model_size)  # Host 端分配
# ... 读取 .om 文件到 buffer ...
model_id = acl.mdl.load_from_mem(model_buffer, model_size)  # 加载到 Device

# 分配输入/输出 Device 内存
inputs[0] = acl.rt.malloc(input_ids_size)    # input_ids
inputs[1] = acl.rt.malloc(mask_size)         # attention_mask
inputs[2] = acl.rt.malloc(pos_ids_size)      # position_ids
inputs[3] = acl.rt.malloc(kv_cache_size)     # KV Cache (常驻 Device!)
outputs[0] = acl.rt.malloc(logits_size)      # logits
outputs[1] = acl.rt.malloc(new_kv_size)      # new_kv_cache
```

---

## 5.1 单步推理执行

```python
def inference(input_data_list, seq_length, is_prefill):
    # 1. Host → Device: 拷贝输入 (不含 KV Cache)
    for i, data in enumerate(input_data_list):  # input_ids, mask, pos_ids
        acl.rt.memcpy(inputs[i]["buffer"], ..., HOST_TO_DEVICE)
    
    # 2. 设置动态 shape 档位 (告诉 NPU 用哪组执行方案)
    acl.mdl.set_input_dynamic_dims(model_id, input_dataset, index, dims)
    
    # 3. 模型执行 (NPU 上完整 forward)
    acl.mdl.execute(model_id, input_dataset, output_dataset)
    
    # 4. 更新 KV Cache (Device → Device, 纯 NPU 内存搬运)
    acl.rt.memcpy(inputs[3] + offset, ..., outputs[1], ..., DEVICE_TO_DEVICE)
    
    # 5. 拷回 logits (仅非 prefill 时)
    if not is_prefill:
        acl.rt.memcpy(outputs[0]["buffer_host"], ..., DEVICE_TO_HOST)
        return logits
    return None
```

---

## 6. KV Cache 管理

### 内存布局 (全部在 Device)

```text
inputs[3] (KV Cache buffer), shape = (1, 4096, 112, 128), dtype=float16
┌───────────────────────────────────────────────────────┐
│ pos 0 │ pos 1 │ ... │ pos N │ ... │ pos 4095         │
│ (有效) │ (有效) │     │(有效) │(空) │ (空)            │
└───────────────────────────────────────────────────────┘
         ← real_kv_size →       ← 未使用 →

每个 position 存储: 112 × 128 = 14336 个 float16 = 28672 bytes
```

---

## 6.1 KV Cache 操作

### Reset (清空)
```python
acl.rt.memset(inputs[3]["buffer"], size, 0, size)  # Device 端置零
real_kv_size = 0
```

### Update (写入新 KV)
```python
# outputs[1] = 本次 forward 产生的 new_kv, shape=(1, seq_len, 112, 128)
# 写入 inputs[3] 的 real_kv_size 偏移位置
acl.rt.memcpy(
    dst = inputs[3] + real_kv_size * 112 * 128 * 2,  # Device 目标地址
    src = outputs[1],                                  # Device 源地址
    kind = DEVICE_TO_DEVICE
)
real_kv_size += seq_len
```

**关键：KV Cache 始终在 Device，不经过 Host，避免 PCIe 瓶颈**

---

## 7. 完整推理演示

### 参数设定

```text
prompt = "你好，世界！"  (tokenize 后 14 个 token)
prompt_length = 14
kv_cache_length = 16
max_prefill_length = 4
half_kv = 16 // 2 = 8
```

### 分解 Prompt

```python
decompose_number(14) → [4, 4, 4, 2]
# 前 3 块 is_prefill=True, 最后 1 块 is_prefill=False
```

---

## 7.1 Prefill 块 0: token[0:4]

**初始状态:** `real_kv_size=0, input_pos=0`

| 数据 | Shape | 值 |
|---|---|---|
| input_ids | (1, 4) | `[tok0, tok1, tok2, tok3]` |
| attention_mask | (1, 12) = (1, 8+4) | `[0,0,0,0,0,0,0,0, 1,1,1,1]` |
| position_ids | (1, 4) | `[0, 1, 2, 3]` |
| kv_cache | (1, **8**, 112, 128) | 全零 (半长档) |

KV 模式: `real_kv_size + seq = 0+4 = 4 ≤ 8` → **半长**

---

## 7.1 Prefill 块 0: 执行过程

```text
[Host CPU]                              [Device NPU]

1. 构造 mask=[0×8, 1×4], pos=[0,1,2,3]
   (Host, numpy 运算)

2. memcpy H2D ─────────────────────→  inputs[0] = input_ids
   memcpy H2D ─────────────────────→  inputs[1] = mask
   memcpy H2D ─────────────────────→  inputs[2] = pos_ids
                                       inputs[3] = KV Cache (已在Device, 不拷贝)

3. set_dynamic_dims(seq=4, kv=8)       选择半长执行方案

4.                                     acl.mdl.execute()
                                       → Embedding → RMSNorm → QKV → RoPE
                                       → Attention(Q@K^T + mask → softmax → @V)
                                       → FFN → ... × 28 layers → LM Head
                                       → outputs[0]=logits(1,4,151936)
                                       → outputs[1]=new_kv(1,4,112,128)

5.                                     update_kv_cache: D2D memcpy
                                       outputs[1] → inputs[3][pos 0:4]
                                       real_kv_size = 4

6. is_prefill=True → 不拷回 logits     (省掉 1.2MB 的 D2H 传输)
```

---

## 7.2 Prefill 块 1: token[4:8]

**状态:** `real_kv_size=4, input_pos=4`

| 数据 | Shape | 说明 |
|---|---|---|
| input_ids | (1, 4) | `[tok4, tok5, tok6, tok7]` |
| attention_mask | (1, 12) | `[1,1,1,1, 0,0,0,0, 1,1,1,1]` |
| position_ids | (1, 4) | `[4, 5, 6, 7]` |
| kv_cache | (1, 8, 112, 128) | pos 0~3 有效，4~7 为零 |

Mask 含义: `[有效KV×4, 空位×4, 当前输入×4]`

判定半长: `4+4=8 ≤ 8` ✅

执行后: `real_kv_size=8`

---

## 7.3 Prefill 块 2: token[8:12]

**状态:** `real_kv_size=8, input_pos=8`

| 数据 | Shape | 说明 |
|---|---|---|
| input_ids | (1, 4) | `[tok8, tok9, tok10, tok11]` |
| attention_mask | (1, 20) = (1, **16**+4) | `[1×8, 0×8, 1×4]` |
| position_ids | (1, 4) | `[8, 9, 10, 11]` |
| kv_cache | (1, **16**, 112, 128) | **切换到全长!** |

判定全长: `8+4=12 > 8` → **全长档**

执行后: `real_kv_size=12`

---

## 7.4 Prefill 块 3 (最后): token[12:14]

**状态:** `real_kv_size=12, input_pos=12`, **is_prefill=False**

| 数据 | Shape | 说明 |
|---|---|---|
| input_ids | (1, 2) | `[tok12, tok13]` |
| attention_mask | (1, 18) | `[1×12, 0×4, 1×2]` |
| position_ids | (1, 2) | `[12, 13]` |
| kv_cache | (1, 16, 112, 128) | pos 0~11 有效 |

**这是最后一块 → 拷回 logits!**

```text
logits shape = (1, 2, 151936)
取 logits[0][-1:] → 采样第 15 个 token
```

执行后: `real_kv_size=14`

---

## 7.5 Decode 阶段: 逐 token 生成

**状态:** `real_kv_size=14, input_pos=14`

每步输入固定:

| 数据 | Shape | 示例 (第 1 步 decode) |
|---|---|---|
| input_ids | (1, 1) | `[sampled_token]` |
| attention_mask | (1, 17) | `[1×14, 0×2, 1×1]` |
| position_ids | (1, 1) | `[14]` |
| kv_cache | (1, 16, 112, 128) | pos 0~13 有效 |

每步:
1. H2D 拷贝 3 个小 tensor
2. `acl.mdl.execute()` → NPU forward
3. D2D 更新 KV Cache (`real_kv_size += 1`)
4. D2H 拷回 logits `(1, 1, 151936)`
5. Host 采样 → 得到下一个 token

---

## 7.6 Attention 中 Mask 的作用

以 Prefill 块 1 为例: mask = `[1,1,1,1, 0,0,0,0, 1,1,1,1]`

```text
Q shape: (1, H, 4, D)      ← 当前 4 个 token 的 query
K_total: (1, H, 12, D)     ← kv_cache(8) + K_new(4) 拼接
scores = Q @ K_total^T → shape (1, H, 4, 12)

外部 mask (padding): [1,1,1,1, 0,0,0,0, 1,1,1,1] → 标记空位
内部 mask (causal):  下三角 → 当前 chunk 内因果约束

合并后每行有效 attend 位置:
  token4: attend to 0,1,2,3 + 自己 = 5 个
  token5: attend to 0,1,2,3,4 + 自己 = 6 个
  token7: attend to 0,...,6 + 自己 = 8 个
```

---

## 7.7 Position IDs 与 RoPE

```python
# RoPE 编码: 将绝对位置信息注入 Q 和 K
Q_rotated = apply_rotary(Q, cos[position_ids], sin[position_ids])
K_rotated = apply_rotary(K, cos[position_ids], sin[position_ids])
```

分块 prefill 时 position_ids 必须正确:
- 块 0: `[0,1,2,3]`
- 块 1: `[4,5,6,7]`（不是 `[0,1,2,3]`!）

这保证 `Q[pos=5] · K[pos=2]` 中包含相对距离 3 的信息，与一次性 forward 数学等价。

---

## 8. 数据流全景图

```text
[Host CPU]                              [Device NPU]
                                        
input_ids (numpy)  ── H2D memcpy ──→  inputs[0]
mask (numpy)       ── H2D memcpy ──→  inputs[1]
pos_ids (numpy)    ── H2D memcpy ──→  inputs[2]
                                      inputs[3] = KV Cache (常驻 Device)
                                          │
                                    acl.mdl.execute()
                                    (完整 Transformer forward)
                                          │
                                      outputs[0] = logits
                                      outputs[1] = new_kv
                                          │
                                    D2D: outputs[1] → inputs[3][offset]
                                    (KV Cache 更新, 不经过 Host)
                                          │
                              ┌─ prefill: 不拷 logits, return None
                              └─ decode:  D2H → logits → Host 采样
```

---

## 8.1 Prefill vs Decode 对比

| | Prefill | Decode |
|---|---|---|
| 输入 token 数 | 多个 (按 2^n 分块) | 1 个 |
| 瓶颈 | Compute-bound | Memory-bound |
| logits 拷回 | 仅最后一块 | 每步都拷 |
| KV Cache 写入 | 每块写入多个位置 | 每步写入 1 个 |
| AI Core 利用率 | 高 (矩阵并行度大) | 低 (M=1, CUBE 空转) |

---

## 9. 性能数据

### Baseline 配置

`kv_cache_length=4096, max_prefill_length=1, Ascend 910`

| 指标 | 数值 |
|---|---|
| TTFT (首字延迟) | 148.92 ms |
| TPOT (每 token) | 8.70 ms |
| Decode 速度 | 114.9 tokens/s |
| 总吞吐量 | 114.6 tokens/s |

### 算子耗时分布 (Profiling)

| 算子 | 占比 | 类型 |
|---|---|---|
| BatchMatMulV2 (Linear) | 58.4% | AI Core, memory-bound |
| StridedSliceD (RoPE) | 19.8% | Vector Core |
| ConcatD (RoPE) | 8.3% | Vector Core |
| Expand (broadcast) | 4.7% | Vector Core |

---

## 10. 总结

### 部署的核心挑战

自回归 LLM 是**动态的** (seq 随生成增长) → 昇腾 NPU 要求**静态的** (预编译固定 shape)

### 解决方案

1. **模型改造**: 固定接口 + KV Cache 张量化 + Eager Attention
2. **ONNX 图修正**: 消除硬件不兼容算子
3. **多档编译**: 有限的 shape 组合覆盖任意输入长度
4. **运行时调度**: 分块 prefill + KV Cache Device 常驻 + 选择性 D2H

### 关键设计选择

- KV Cache 全程在 Device → 避免 Host↔Device 数据搬运瓶颈
- Prefill 跳过 logits 拷贝 → 省掉无用的 D2H 传输
- 半长/全长 KV 两档 → 前期 attention 计算量减半

---

## 附录: 关键代码位置

| 功能 | 文件 | 行号 |
|---|---|---|
| 模型静态化 | `export/modeling_qwen2.py` | 760 |
| KV Cache 切片 | `export/modeling_qwen2.py` | 283-290 |
| ONNX 导出 | `export/export_onnx.py` | 213-224 |
| 图修正 | `export/change_node.py` | 50-68 |
| ATC 编译 | `export/onnx2om.py` | 181-213 |
| ACL 初始化 | `utils/engine.py` | 54-64 |
| 内存分配 | `utils/engine.py` | 270-314 |
| 推理执行 | `utils/engine.py` | 333-470 |
| KV Cache 更新 | `utils/engine.py` | 162-190 |
| Mask 构造 | `utils/engine.py` | 117-148 |
| Prefill 分块 | `utils/session.py` | 181-232 |
| Token 生成循环 | `utils/inference.py` | 195-240 |
