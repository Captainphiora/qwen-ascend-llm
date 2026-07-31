#!/bin/bash
# ============================================================
# 采样策略性能测试 & Profiling 一键脚本
# 用法:
#   bash scripts/bench_sampling.sh              # 运行全部测试
#   bash scripts/bench_sampling.sh --bench-only # 仅性能对比
#   bash scripts/bench_sampling.sh --prof-only  # 仅 profiling
#   bash scripts/bench_sampling.sh --msprof     # 带 ACL profiling 原始数据采集
# ============================================================

set -e

# ---- 配置参数 (按需修改) ----
DEVICE_ID=7
OM_MODEL_PATH="output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om"
HF_MODEL_DIR="/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
PROMPT="请详细介绍一下机器学习的基本概念和常用算法"
MAX_NEW_TOKENS=100
ROUNDS=3
WARMUP=1
PROFILING_DIR="./profiling_sampling_data"
# ---- 配置结束 ----

# 加载 CANN 环境
source ~/.bashrc_cann900

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="benchmark_results"
mkdir -p "$RESULT_DIR"

# 解析参数
RUN_BENCH=true
RUN_PROF=true
USE_MSPROF=""

for arg in "$@"; do
    case "$arg" in
        --bench-only) RUN_PROF=false ;;
        --prof-only)  RUN_BENCH=false ;;
        --msprof)     USE_MSPROF="--use_msprof" ;;
        --device_id=*) DEVICE_ID="${arg#*=}" ;;
        --help|-h)
            echo "用法: bash scripts/bench_sampling.sh [OPTIONS]"
            echo ""
            echo "OPTIONS:"
            echo "  --bench-only    仅运行性能对比 benchmark"
            echo "  --prof-only     仅运行详细 profiling"
            echo "  --msprof        同时采集 ACL profiling 原始数据"
            echo "  --device_id=N   指定 NPU 设备号 (默认 7)"
            echo "  --help          显示帮助"
            exit 0
            ;;
    esac
done

echo "============================================================"
echo " 采样策略性能测试"
echo " Device: npu:${DEVICE_ID}"
echo " Model:  ${OM_MODEL_PATH}"
echo " Time:   ${TIMESTAMP}"
echo "============================================================"
echo ""

COMMON_ARGS="--device_id $DEVICE_ID \
    --om_model_path $OM_MODEL_PATH \
    --hf_model_dir $HF_MODEL_DIR \
    --kv_cache_length $KV_CACHE_LENGTH \
    --max_prefill_length $MAX_PREFILL_LENGTH \
    --prompt \"$PROMPT\" \
    --max_new_tokens $MAX_NEW_TOKENS"

# ---- Part 1: 性能对比 Benchmark ----
if [ "$RUN_BENCH" = true ]; then
    echo ">>> [1/2] 运行采样策略性能对比..."
    BENCH_OUTPUT="${RESULT_DIR}/sampling_bench_${TIMESTAMP}.txt"

    eval python benchmark_sampling.py \
        $COMMON_ARGS \
        --rounds $ROUNDS \
        --warmup $WARMUP \
        2>&1 | tee "$BENCH_OUTPUT"

    echo ""
    echo ">>> Benchmark 结果已保存: $BENCH_OUTPUT"
    echo ""
fi

# ---- Part 2: 详细 Profiling ----
if [ "$RUN_PROF" = true ]; then
    echo ">>> [2/2] 运行详细 Profiling..."
    PROF_OUTPUT="${RESULT_DIR}/sampling_profile_${TIMESTAMP}.txt"

    eval python profile_sampling.py \
        $COMMON_ARGS \
        $USE_MSPROF \
        --profiling_dir "$PROFILING_DIR" \
        2>&1 | tee "$PROF_OUTPUT"

    echo ""
    echo ">>> Profiling 结果已保存: $PROF_OUTPUT"

    if [ -n "$USE_MSPROF" ] && [ -d "$PROFILING_DIR" ]; then
        echo ""
        echo ">>> ACL Profiling 原始数据: $PROFILING_DIR"
        echo "    解析命令: msprof --export=on --output=$PROFILING_DIR/"
        echo "    可视化:   用 MindStudio 打开 $PROFILING_DIR/ 下的 .prof 文件"
    fi
    echo ""
fi

echo "============================================================"
echo " 测试完成! 结果文件:"
[ "$RUN_BENCH" = true ] && echo "   - $BENCH_OUTPUT"
[ "$RUN_PROF" = true ] && echo "   - $PROF_OUTPUT"
echo "============================================================"
