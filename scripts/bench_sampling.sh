#!/bin/bash
# ============================================================
# 采样策略性能测试 & 完整 Profiling 一键脚本
#
# 包含:
#   Part 1: 采样策略性能对比 (greedy/top_p/top_k × NPU/CPU)
#   Part 2: 详细 Profiling (Host/Device 耗时, 数据搬运, 算力利用率)
#   Part 3: ACL 算子级 Profiling (采集+解析, 各算子耗时统计)
#
# 用法:
#   bash scripts/bench_sampling.sh              # 运行全部 (Part 1+2+3)
#   bash scripts/bench_sampling.sh --bench-only # 仅 Part 1
#   bash scripts/bench_sampling.sh --prof-only  # 仅 Part 2+3
#   bash scripts/bench_sampling.sh --no-acl     # 跳过 Part 3 (ACL算子profiling)
#   bash scripts/bench_sampling.sh --device_id=5
#   bash scripts/bench_sampling.sh --device_id=5 --npu-sampling
# ============================================================

set -e

# ---- 配置参数 (按需修改) ----
DEVICE_ID=6
# OM_MODEL_PATH="output/om_v4_noexpand/DeepSeek-R1-Distill-Qwen-1.5B_4096_8.om"
# OM_MODEL_PATH="opt_models/v4_noexpand/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand.om"
# OM_MODEL_PATH="output/no_rope/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_no_rope.om"
OM_MODEL_PATH="output/no_rope/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_no_rope_sim.om"
# OM_MODEL_PATH="opt_models/v0_baseline/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v0_baseline.om"
HF_MODEL_DIR="../models/DeepSeek-R1-Distill-Qwen-1.5B"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
PROMPT="请详细介绍一下机器学习的基本概念和常用算法"
MAX_NEW_TOKENS=100
ROUNDS=3
WARMUP=1
PROFILING_DIR="./profiling_sampling_data"
PROF_MAX_TOKENS=20   # ACL profiling 仅采集少量 token (减少数据量)
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
RUN_ACL_PROF=true
USE_NPU_SAMPLING=false

for arg in "$@"; do
    case "$arg" in
        --bench-only) RUN_PROF=false; RUN_ACL_PROF=false ;;
        --prof-only)  RUN_BENCH=false ;;
        --no-acl)     RUN_ACL_PROF=false ;;
        --npu-sampling) USE_NPU_SAMPLING=true ;;
        --device_id=*) DEVICE_ID="${arg#*=}" ;;
        --help|-h)
            echo "用法: bash scripts/bench_sampling.sh [OPTIONS]"
            echo ""
            echo "OPTIONS:"
            echo "  --bench-only    仅运行性能对比 benchmark (Part 1)"
            echo "  --prof-only     仅运行 profiling (Part 2+3)"
            echo "  --no-acl        跳过 ACL 算子级 profiling (Part 3)"
            echo "  --npu-sampling  启用 NPU ATB 采样对比 (会有退出时的 harmless warning)"
            echo "  --device_id=N   指定 NPU 设备号 (默认 7)"
            echo "  --help          显示帮助"
            exit 0
            ;;
    esac
done

# 设置 NPU 采样环境变量
if [ "$USE_NPU_SAMPLING" = true ]; then
    export USE_NPU_SAMPLING=1
else
    export USE_NPU_SAMPLING=0
fi

# msprof 路径 (CANN tools 目录)
MSPROF="${ASCEND_TOOLKIT_HOME}/tools/profiler/bin/msprof"
if [ ! -f "$MSPROF" ]; then
    MSPROF=$(find /usr/local/Ascend -name "msprof" -path "*/cann-9*/bin/msprof" 2>/dev/null | head -1)
fi

OUTPUT_FILE="${RESULT_DIR}/sampling_full_report_${TIMESTAMP}.txt"

echo "============================================================" | tee "$OUTPUT_FILE"
echo " 采样策略完整性能测试 & Profiling 报告" | tee -a "$OUTPUT_FILE"
echo " Device: npu:${DEVICE_ID}" | tee -a "$OUTPUT_FILE"
echo " Model:  ${OM_MODEL_PATH}" | tee -a "$OUTPUT_FILE"
echo " Time:   ${TIMESTAMP}" | tee -a "$OUTPUT_FILE"
echo "============================================================" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

COMMON_ARGS="--device_id $DEVICE_ID \
    --om_model_path $OM_MODEL_PATH \
    --hf_model_dir $HF_MODEL_DIR \
    --kv_cache_length $KV_CACHE_LENGTH \
    --max_prefill_length $MAX_PREFILL_LENGTH"

# ---- Part 1: 性能对比 Benchmark ----
if [ "$RUN_BENCH" = true ]; then
    echo ">>> [Part 1/3] 采样策略性能对比 (${ROUNDS} rounds × 8 configs)..." | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"

    eval python benchmark_sampling.py \
        $COMMON_ARGS \
        --prompt \"$PROMPT\" \
        --max_new_tokens $MAX_NEW_TOKENS \
        --rounds $ROUNDS \
        --warmup $WARMUP \
        2>&1 | tee -a "$OUTPUT_FILE"

    echo "" | tee -a "$OUTPUT_FILE"
fi

# ---- Part 2: 详细 Profiling (Host/Device/H2D/算力) ----
if [ "$RUN_PROF" = true ]; then
    echo ">>> [Part 2/3] 详细 Profiling (Host/Device/数据搬运/算力利用率)..." | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"

    eval python profile_sampling.py \
        $COMMON_ARGS \
        --prompt \"$PROMPT\" \
        --max_new_tokens $MAX_NEW_TOKENS \
        2>&1 | tee -a "$OUTPUT_FILE"

    echo "" | tee -a "$OUTPUT_FILE"
fi

# ---- Part 3: ACL 算子级 Profiling (采集 + 解析) ----
if [ "$RUN_ACL_PROF" = true ]; then
    echo ">>> [Part 3/3] ACL 算子级 Profiling (采集 ${PROF_MAX_TOKENS} tokens)..." | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"

    # 清理旧 profiling 数据
    rm -rf "$PROFILING_DIR"
    mkdir -p "$PROFILING_DIR"

    if [ -z "$MSPROF" ] || [ ! -f "$MSPROF" ]; then
        echo "[ERROR] msprof 未找到, 无法采集 ACL profiling" | tee -a "$OUTPUT_FILE"
    else
        # 使用 msprof 包裹 Python 执行 (采集完整 host + device 数据)
        echo ">>> 使用 msprof 包裹执行采集 (host + device 全量数据)..." | tee -a "$OUTPUT_FILE"
        echo ">>> msprof: $MSPROF" | tee -a "$OUTPUT_FILE"
        echo "" | tee -a "$OUTPUT_FILE"

        eval "$MSPROF" --output="$PROFILING_DIR" \
            --ascendcl=on \
            --task-time=on \
            --ai-core=on \
            --aic-metrics=PipeUtilization \
            --aicpu=on \
            --runtime-api=on \
            python profile_sampling.py \
                $COMMON_ARGS \
                --prompt \"$PROMPT\" \
                --max_new_tokens $PROF_MAX_TOKENS \
                --sampling_method greedy \
            2>&1 | tee -a "$OUTPUT_FILE"

        echo "" | tee -a "$OUTPUT_FILE"

        # 解析: msprof --export
        PROF_SUBDIR=$(find "$PROFILING_DIR" -maxdepth 1 -type d -name "PROF_*" | sort | tail -1)
        if [ -n "$PROF_SUBDIR" ]; then
            echo ">>> 解析 Profiling 数据: $PROF_SUBDIR" | tee -a "$OUTPUT_FILE"
            "$MSPROF" --export=on --output="$PROF_SUBDIR" 2>&1 | tee -a "$OUTPUT_FILE" || true
        fi

        echo "" | tee -a "$OUTPUT_FILE"

        # 查找并展示 CSV
        echo ">>> 解析后的 CSV 文件:" | tee -a "$OUTPUT_FILE"
        find "$PROFILING_DIR" -name "*.csv" 2>/dev/null | sort | tee -a "$OUTPUT_FILE"
        echo "" | tee -a "$OUTPUT_FILE"

        # 展示算子统计
        OP_STAT=$(find "$PROFILING_DIR" -name "op_statistic*.csv" 2>/dev/null | sort | tail -1)
        if [ -n "$OP_STAT" ] && [ -f "$OP_STAT" ]; then
            echo ">>> 算子耗时统计: $OP_STAT" | tee -a "$OUTPUT_FILE"
            echo "┌── Device 算子耗时统计 ──────────────────────────────────────────┐" | tee -a "$OUTPUT_FILE"
            head -21 "$OP_STAT" | column -t -s',' 2>/dev/null | tee -a "$OUTPUT_FILE" || head -21 "$OP_STAT" | tee -a "$OUTPUT_FILE"
            echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$OUTPUT_FILE"
        fi

        # 展示 Host API 统计
        API_STAT=$(find "$PROFILING_DIR" -name "api_statistic*.csv" 2>/dev/null | sort | tail -1)
        if [ -n "$API_STAT" ] && [ -f "$API_STAT" ]; then
            echo "" | tee -a "$OUTPUT_FILE"
            echo ">>> Host API 耗时统计: $API_STAT" | tee -a "$OUTPUT_FILE"
            echo "┌── Host 侧 ACL API 耗时统计 ────────────────────────────────────┐" | tee -a "$OUTPUT_FILE"
            head -21 "$API_STAT" | column -t -s',' 2>/dev/null | tee -a "$OUTPUT_FILE" || head -21 "$API_STAT" | tee -a "$OUTPUT_FILE"
            echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$OUTPUT_FILE"

            # 提取数据搬运 (H2D/D2H) 相关统计
            echo "" | tee -a "$OUTPUT_FILE"
            echo "┌── 数据搬运 (H2D/D2H) 耗时统计 ─────────────────────────────────┐" | tee -a "$OUTPUT_FILE"
            echo "  (从 api_statistic 中提取 Memcpy/Copy 相关条目)" | tee -a "$OUTPUT_FILE"
            echo "" | tee -a "$OUTPUT_FILE"
            # 表头
            head -1 "$API_STAT" | column -t -s',' 2>/dev/null | tee -a "$OUTPUT_FILE" || head -1 "$API_STAT" | tee -a "$OUTPUT_FILE"
            # Memcpy 相关行
            grep -i "memcpy\|MemCopy\|InputCopy\|OutputCopy\|Copy" "$API_STAT" | column -t -s',' 2>/dev/null | tee -a "$OUTPUT_FILE" || grep -i "memcpy\|MemCopy\|InputCopy\|OutputCopy\|Copy" "$API_STAT" | tee -a "$OUTPUT_FILE"
            echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$OUTPUT_FILE"
        fi

        echo "" | tee -a "$OUTPUT_FILE"
        echo ">>> Profiling 原始数据: $PROFILING_DIR" | tee -a "$OUTPUT_FILE"
        echo "    可视化: 用 MindStudio 打开 $PROF_SUBDIR" | tee -a "$OUTPUT_FILE"
    fi

    echo "" | tee -a "$OUTPUT_FILE"
fi

echo "============================================================" | tee -a "$OUTPUT_FILE"
echo " 测试完成!" | tee -a "$OUTPUT_FILE"
echo " 完整报告: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "============================================================" | tee -a "$OUTPUT_FILE"
