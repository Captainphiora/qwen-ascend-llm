# AGENTS.md — 项目规则与硬件上下文

## 项目概述

本项目在 **Atlas 200I A2 (310B1)** 嵌入式开发板上，以 **PyTorch → ONNX → OM** 流程部署
DeepSeek-R1-Distill-Qwen-1.5B 大模型推理，目标是在该设备上达到 **最大吞吐 / 最小时延**。

## 硬件环境（实测快照）

| 项目 | 值 |
|------|-----|
| 设备 | Atlas 200I A2 开发者套件 (IT22MMDA) |
| NPU 芯片 | Ascend 310B1, 1× AI Core |
| 算力档位 | **20T_1.6GHz**（当前生效） |
| FP16 算力 | 10 TFLOPS |
| INT8 算力 | 20 TOPS |
| CPU | AArch64 4核 @ 1.6GHz |
| 精度支持 | FP16、INT8；**不支持 BF16** |

### 内存架构（统一共享内存）

本设备 **CPU 和 NPU 共享同一块 12 GB LPDDR4X 物理内存**，没有独立显存。
`npu-smi` 报告的"显存占用"实际上是**整个系统的内存占用**（内核 + 所有用户进程 + 文件缓存），
与 `free` 报告的是同一块物理内存。两者关系为：`NPU used ≈ 11577 - MemAvailable`。

清理后 npu-smi 仍显示 ~4 GB 占用，构成如下（不可避免）：
- VSCode Remote Server: ~3.2 GB（最大消费者）
- Linux 内核 + 驱动: ~200 MB
- Ascend 守护进程 (davinci service): ~76 MB
- Kerminal + 其他: ~300 MB
- 文件缓存: ~500 MB（可回收）

> **结论**：npu-smi 的"显存"没有被 NPU 独占，推理可用内存 = 物理总量 - 系统开销。
> 当前系统固定开销约 3.3 GB（含 VSCode Remote ~2.1 GB，Pylance 和 Copilot 已禁用），
> 留 512 MB 安全余量后，推理可用上限为 **7.7 GB**。

| 项目 | 值 |
|------|-----|
| 物理内存总量 | 12 GB LPDDR4X |
| 内存带宽 | 51.2 GB/s（96-bit @ 2131 MHz） |
| NPU 可寻址 | 11577 MB（= Linux MemTotal，同一块内存） |
| 系统基础开销 | ~3.7 GB（内核 + VSCode + 守护进程） |
| SWAP | 8 GB（/swapfile），测试期间禁止使用 |
| 存储 | SD 卡 114G（可用 17G）+ eMMC 57G（挂载 /mnt/emmc，可用 34G） |

## 软件环境

| 组件 | 值 |
|------|------|
| OS | Ubuntu 22.04 / EulerOS kernel 6.6.0 aarch64 |
| CANN | 9.0.0（主用） |
| CANN 环境初始化 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| NPU 驱动 | 26.0.rc1 |
| MindIE | 2.3.0（mindie-llm / mindie-service） |
| Conda 环境 | **msit_compare**（须先 `conda activate msit_compare`） |

### 环境激活顺序

```bash
conda activate msit_compare
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 可用 OM 模型

路径：`output/model/`（符号链接到 `/mnt/emmc/model/`）

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om` | 3.4G | FP16 v4优化版，KV长度4096 |

## 当前测试目标

**在 310B1 上跑出 1.5B 模型 OM 部署的最大吞吐 / 最小时延。**

### 资源硬限制

| 约束 | 限制值 | 说明 |
|------|--------|------|
| 推理可用内存 | **7.7 GB (7885 MB)** | 物理总量 11577 MB - 系统固定开销 ~3300 MB - 安全余量 512 MB |
| SWAP 使用 | **禁止** | 测试期间 SWAP 使用量必须为 0，触及 SWAP 视为测试失败 |

> 资源预算估算：推理可用 7.7 GB - 模型权重 3.4 GB = **~4.3 GB** 给 KV Cache + 推理中间缓冲。
> 系统固定开销 ~3.3 GB（含 VSCode Remote ~2.1 GB、内核 + 驱动 ~0.7 GB、Kerminal + 守护进程 ~0.5 GB），
> 已实测稳定（波动 ±15 MB），Pylance 和 GitHub Copilot 已禁用。

### 关键约束

- 统一共享内存架构，NPU 分配会直接挤压 CPU 可用内存
- Decode 阶段是**内存带宽瓶颈**（51.2 GB/s），FP16 理论上限 ~17 tok/s
- 单 AI Core，算力天花板低，Prefill 阶段是计算瓶颈
- CPU 仅 4 核 1.6GHz ARM，tokenizer/采样等 host 侧开销需关注

### 优化方向优先级

1. **减少 Host-Device 数据搬运** — 尽量在 NPU 侧完成采样
2. **KV Cache 长度调优** — 在 10 GB 显存预算内找最优值
3. **Prefill 分块策略** — 适配单 AI Core 算力
4. **Profiling 驱动** — 用 msprof 定位实际瓶颈，不猜测

## 目录结构

```
├── config.py                  # 推理配置类
├── main.py / api.py / server.py / cli_chat.py  # 入口和服务
├── utils/                     # 推理引擎 (engine/session/kvcache/inference)
├── export/                    # PyTorch→ONNX→OM 导出代码
├── configs/                   # 运行配置 JSON
├── models/                    # HF 模型权重
├── output/                    # 产出物 (onnx/om)，符号链接到 /mnt/emmc
├── scripts/                   # 可执行脚本
│   ├── export/                #   导出脚本
│   ├── profiling/             #   Profiling 脚本
│   ├── benchmark/             #   基准测试脚本
│   ├── deploy/                #   部署/服务脚本
│   └── docker/                #   Docker 脚本
├── profiling/                 # Profiling 代码和数据
├── benchmarks/                # 基准测试代码和结果
├── tests/                     # 测试
├── docs/                      # 文档
├── assets/                    # 图片资源
├── client/                    # OpenAI API 客户端示例
├── logs/                      # 运行日志
└── opt_logs/                  # 优化记录
```

## 编码规范

- Python 代码须兼容 AArch64 + CANN 9.0 环境
- 所有路径使用相对路径或 `os.path.join(project_dir, ...)`，不硬编码绝对路径
- OM 模型文件存放于 `output/model/`，通过 configs/*.json 配置引用
- Profiling 数据存放于 `profiling/` 目录下
- Shell 脚本统一放在 `scripts/` 的对应子目录中
- 新增 Python 文件若在子目录中需导入根模块，使用 `sys.path.insert(0, project_root)` 模式
- 性能数据必须标注测试条件（模型版本、KV长度、输入长度、算力档位）
- 不提交 `kernel_meta/`、`__pycache__/`、`*.om`、profiling 原始数据等生成物
