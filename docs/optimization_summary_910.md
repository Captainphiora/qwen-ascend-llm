# 910 优化总结：做了什么、哪些 work、哪些没 work、接下来怎么办

> 日期: 2026-09-07
> 基线: v5_gate_up_fuse on Ascend 910_9382, batch=1, kv_cache=4096, decode
> 基线性能: 6.15 ms/tok, 163 tok/s

---

## 1. 目标

减少中间数据搬运量、提高带宽利用率，从而提升 batch=1 decode 推理性能。

## 2. 做了什么

### 2.1 搬运量分析（发现问题）

对 v5 的 profiling 做了搬运量拆解，用 `Σ(Input_Shapes + Output_Shapes)` 计算每 token 总搬运量。
初始分析得到 7.98 GB/tok，其中 KV cache 搬运 4.75 GB（60%）——结论是 KV cache 搬运是主要瓶颈。

### 2.2 v7: KV cache 按层预切分（31 输入）

把 1 个 `[1,112,4096,128]` 共享 KV 张量拆为 28 个独立 `[1,4,4096,128]`。
OM 模型从 4 输入变为 31 输入。

### 2.3 v7b: 图内 Split（保持 4 输入）

保持 v6 的 4 输入接口，在 ONNX 图内用 `torch.chunk(28, dim=1)` 一次性切分。

### 2.4 v8: QKV 投影合并

把 q_proj + k_proj + v_proj 三次 MatMul 合并为一次 qkv_proj MatMul。

### 2.5 HBM 带宽微基准

独立测试了 910 的实际可达带宽：大块拷贝 ~1180 GB/s、各尺寸 MatMul 的带宽。

## 3. 结果汇总

| 版本 | 改动 | ONNX 节点 | BMM ms/tok | 总 TPOT | 吞吐 | 结果 |
|------|------|----------|-----------|---------|------|------|
| **v5** | 基线 (gate_up 合并) | 6124 | 4.55 | 6.15 ms | **163 tok/s** | 基线 |
| **v6** | BHSD layout | 6122 | 4.52 | 6.27 ms | 160 tok/s | ≈ 持平 |
| v7 | 31 输入 per-layer KV | 6626 | 9.84 | 15.53 ms | 64 tok/s | ❌ BMM 2× 退化 |
| v7b | 图内 Split | 6691 | 4.55 | 8.76 ms | 114 tok/s | BMM OK, Slice 慢 |
| v8 | QKV 合并 | 6711 | 9.12 | 13.07 ms | ~77 tok/s | ❌ BMM 2× 退化 |
| **v10** | W8A8 + gate_up prefuse | — | 4.03(Q+F) | 5.79 ms* | **173 tok/s*** | ✅ (*不同采集条件) |
| **v10 当前环境** | 同上 (原始 OM 重测) | — | — | 10.34 ms | 97 tok/s | wall-clock Device 15 |
| **v5+QKV FP16** | QKV ONNX合并 | 6264 | — | 10.86 ms | 92 tok/s | ✅ |
| **v5+QKV W8A8** | QKV合并 + AMCT | — | — | **10.08 ms** | **99 tok/s** | ✅ **当前最优** |
| **v11** | KV in-place (where) | 5968 | 4.45 | 6.05 ms | **165 tok/s** | ✅ 小幅改善 |
| **v11+QKV** | + change_node QKV合并 | 5800 | 3.84 | 5.48 ms | **183 tok/s** | ✅ FP16最优 |
| v11 W8A8 | AMCT on BHSD图 | 6203 | 2.81(Q)+1.19 | 12.80 ms | 78 tok/s | ❌ BHSD Vector退化 |

## 4. 哪些 work 了

1. **搬运量分析框架建立** — `calc_theoretical_throughput.py` + profiling 全流程
2. **HBM 带宽微基准** — 确认硬件可达 ~1180 GB/s，大矩阵 BMM 已接近极限
3. **v7 的搬运量确实降了** — 声明搬运量从 7.98 GB 降到 4.34 GB（-45%）
4. **环境问题定位和封装** — CANN 路径陷阱、ONNX 导出 dtype 问题，写入 AGENTS.md
5. **v10 W8A8 量化** — 权重 MatMul 从 3.96 ms 降到 ~2.8 ms，总 TPOT 5.79 ms (173 tok/s)
6. **v11 KV in-place** — ConcatD 消除 + GatherV2 减少，TPOT 6.05 ms (165 tok/s, FP16 最优)

## 5. 哪些没 work

**所有改变 ONNX 图拓扑的优化都触发了 ATC 编译器全局退化**：

| 尝试 | 新增节点 | BMM 退化 | 根因 |
|------|---------|---------|------|
| v7 (31 输入) | +504 | 2.2× | 多输入导致 ATC 内存布局策略变化 |
| v7b (Split) | +569 | 无 | BMM OK，但 StridedSliceD 本身很慢 |
| v8 (QKV 合并) | +589 | 2.0× | 140 个 Slice 节点改变图拓扑 |
| v11 ScatterElements | — | 无 | ScatterElements 落到 AI_CPU (305μs/call) |

**关键发现**：
- v6 的 6122 个节点是 ATC 编译器的 sweet spot。节点数增加 500+ 可能触发退化。
- v11 证明: 图拓扑变化 + 节点数变化，只要 input/output 数不变 (仍为 4+2)，BMM 不退化。
  这修正了之前"所有图拓扑变化都触发退化"的结论——退化与 input/output 数量更相关。
- ScatterElements 在 CANN 9.0.0 上无 NPU kernel，必须用 Where(SelectV2) 替代。

### 搬运量分析方法论错误

最初的"KV cache 搬运 4.75 GB"是**高估的**。v7b 排查证明 GatherV2 实际只读取
contiguous 子块（~4 MB），不是声明的整个张量（115 MB）。
v5 的 KV cache 实际开销只有 **0.81 ms（13% of TPOT）**，不是主要瓶颈。

## 6. 当前难点

### 6.1 ATC 编译器对图拓扑极其敏感

这是最大的障碍。三次不同方向的尝试（多输入、Split、QKV 合并）都因为增加 ONNX 节点
而触发全局退化。这不是某个算子的问题，是编译器在看到"不同"的图结构时，对所有权重的
HBM 访问模式做了更保守的调度。

### 6.2 v5/v6 已经接近当前框架的最优

| 组件 | v5 耗时 | 占比 | 优化空间 |
|------|--------|------|---------|
| BMM (权重读取+计算) | 4.55 ms | 74% | 受限于 batch=1 的低算术强度 |
| KV cache (Gather+Concat) | 0.81 ms | 13% | 已经高效 (GatherV2 做了子块优化) |
| 其他 (RoPE/Add/Softmax) | 0.79 ms | 13% | 空间有限 |

BMM 的 4.55 ms 中，权重搬运 ~3.09 GB 在 ~680 GB/s 有效带宽下需要 ~4.5 ms，
几乎全是带宽消耗。batch=1 时算术强度只有 1.0 FLOPs/Byte（拐点 235），
376 TFLOPS 的算力只用了 0.13%。

### 6.3 OM 框架本身的限制

当前的 PyTorch → ONNX → ATC → OM 流程，编译器的图优化是黑盒。
无法控制 ATC 的内存布局策略、算子融合决策、kernel 调度方式。

## 7. 接下来的方向

### 方向 A: ONNX 图级别优化（低风险）

1. **QKV MatMul 合并 (change_node 层面)**:
   把三个 MatMul (q/k/v_proj) 权重 concat 为 [1536, 2048]，用 1 个 MatMul + Split 替换。
   节点数减少 2 个，不增加 input/output，不触发 ATC 退化。
   v11 数据: q(0.303)+k(0.270)+v(0.272)=0.845ms → 合并后 ~0.40ms，节省 **0.45ms (7.4%)**。

2. **清理残余 RoPE ConcatD**: v11 仍有 1215 个 ConcatD (`1,12,1,64;1,12,1,64`).
   检查 change_node 模式匹配，确保所有 RoPE 都被 NPURotaryPosEmb 替换。
   节省 **~0.16ms (2.6%)**。

3. **v11 + W8A8 组合**: 将 v11 KV in-place 应用到 W8A8 模型。预计 ~5.74ms。

### 方向 B: 自定义算子（中风险）

4. **AscendC KVCacheUpdate kernel**: 替代 SelectV2+Equal (0.386ms, 85 kernel/tok)，
   用单个 kernel 做 buffer copy + position overwrite。预计节省 **~0.34ms**。

### 方向 C: 换推理框架（中风险，高收益）

5. **MindIE / vLLM-Ascend**：本机已装 MindIE 2.3.0。内置 FlashAttention、
   PagedAttention、连续批处理等优化。batch=1 预期 3-4 ms/tok；batch=8+ <1 ms/tok。

### 建议优先级

**A1（QKV 合并）→ A2（RoPE 清理）→ A3（+W8A8）→ B（自定义 kernel）→ C（MindIE）**

A1+A2 预计从 6.05ms 降到 5.44ms (FP16, -10%)；叠加 W8A8 约 5.0ms (-14%)。
详细分析见 `work_logs/v11_optimization_directions.md`。
