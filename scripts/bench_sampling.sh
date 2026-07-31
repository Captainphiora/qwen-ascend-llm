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

for arg in "$@"; do
    case "$arg" in
        --bench-only) RUN_PROF=false; RUN_ACL_PROF=false ;;
        --prof-only)  RUN_BENCH=false ;;
        --no-acl)     RUN_ACL_PROF=false ;;
        --device_id=*) DEVICE_ID="${arg#*=}" ;;
        --help|-h)
            echo "用法: bash scripts/bench_sampling.sh [OPTIONS]"
            echo ""
            echo "OPTIONS:"
            echo "  --bench-only    仅运行性能对比 benchmark (Part 1)"
            echo "  --prof-only     仅运行 profiling (Part 2+3)"
            echo "  --no-acl        跳过 ACL 算子级 profiling (Part 3)"
            echo "  --device_id=N   指定 NPU 设备号 (默认 7)"
            echo "  --help          显示帮助"
            exit 0
            ;;
    esac
done

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
        2>&1 | grep -v "EE9999\|107002\|107003\|ctx is NULL\|context is empty\|function operator\|StreamDestroy\|DeviceSynchronize\|npuSynchronize\|TraceBack\|compiler_depend\|rtGetDevMsg\|Check whether\|Solution: 1" | tee -a "$OUTPUT_FILE"

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
        2>&1 | grep -v "EE9999\|107002\|107003\|ctx is NULL\|context is empty\|function operator\|StreamDestroy\|DeviceSynchronize\|npuSynchronize\|TraceBack\|compiler_depend\|rtGetDevMsg\|Check whether\|Solution: 1" | tee -a "$OUTPUT_FILE"

    echo "" | tee -a "$OUTPUT_FILE"
fi

# ---- Part 3: ACL 算子级 Profiling (采集 + 解析) ----
if [ "$RUN_ACL_PROF" = true ]; then
    echo ">>> [Part 3/3] ACL 算子级 Profiling (采集 ${PROF_MAX_TOKENS} tokens)..." | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"

    # 清理旧 profiling 数据
    rm -rf "$PROFILING_DIR"
    mkdir -p "$PROFILING_DIR"

    # 使用已有的 profile_inference.py 的逻辑, 通过 profile_sampling.py --use_msprof
    eval python profile_sampling.py \
        $COMMON_ARGS \
        --prompt \"$PROMPT\" \
        --max_new_tokens $PROF_MAX_TOKENS \
        --sampling_method greedy \
        --use_msprof \
        --profiling_dir "$PROFILING_DIR" \
        2>&1 | grep -v "EE9999\|107002\|107003\|ctx is NULL\|context is empty\|function operator\|StreamDestroy\|DeviceSynchronize\|npuSynchronize\|TraceBack\|compiler_depend\|rtGetDevMsg\|Check whether\|Solution: 1" | tee -a "$OUTPUT_FILE"

    echo "" | tee -a "$OUTPUT_FILE"

    # 解析 profiling 数据
    if [ -d "$PROFILING_DIR" ] && [ "$(ls -A $PROFILING_DIR 2>/dev/null)" ]; then
        echo ">>> 解析 ACL Profiling 数据..." | tee -a "$OUTPUT_FILE"

        # 使用 msprof 导出
        MSPROF_EXPORT_DIR="${PROFILING_DIR}/export_${TIMESTAMP}"
        msprof --export=on --output="$PROFILING_DIR" \
            --export-path="$MSPROF_EXPORT_DIR" 2>&1 | tee -a "$OUTPUT_FILE" || true

        # 查找并解析算子统计 CSV
        OP_SUMMARY=""
        for csv_file in $(find "$MSPROF_EXPORT_DIR" -name "op_statistic*.csv" 2>/dev/null | head -1); do
            OP_SUMMARY="$csv_file"
        done

        if [ -z "$OP_SUMMARY" ]; then
            # 尝试其他路径
            for csv_file in $(find "$PROFILING_DIR" -name "op_statistic*.csv" 2>/dev/null | head -1); do
                OP_SUMMARY="$csv_file"
            done
        fi

        if [ -n "$OP_SUMMARY" ] && [ -f "$OP_SUMMARY" ]; then
            echo "" | tee -a "$OUTPUT_FILE"
            echo "┌── 算子耗时统计 (Top 20) ────────────────────────────────────────┐" | tee -a "$OUTPUT_FILE"
            head -21 "$OP_SUMMARY" | column -t -s',' 2>/dev/null | tee -a "$OUTPUT_FILE" || head -21 "$OP_SUMMARY" | tee -a "$OUTPUT_FILE"
            echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$OUTPUT_FILE"
        else
            echo "[WARN] 未找到 op_statistic CSV, 尝试列出可用文件:" | tee -a "$OUTPUT_FILE"
            find "$PROFILING_DIR" -name "*.csv" -o -name "*.json" 2>/dev/null | head -20 | tee -a "$OUTPUT_FILE"

            # 尝试直接输出 summary
            SUMMARY_FILE=$(find "$PROFILING_DIR" -name "*summary*" -name "*.csv" 2>/dev/null | head -1)
            if [ -n "$SUMMARY_FILE" ]; then
                echo "" | tee -a "$OUTPUT_FILE"
                echo "┌── 算子摘要 ────────────────────────────────────────────────────┐" | tee -a "$OUTPUT_FILE"
                head -30 "$SUMMARY_FILE" | tee -a "$OUTPUT_FILE"
                echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$OUTPUT_FILE"
            fi
        fi

        echo "" | tee -a "$OUTPUT_FILE"
        echo ">>> ACL Profiling 原始数据: $PROFILING_DIR" | tee -a "$OUTPUT_FILE"
        echo "    可视化: 用 MindStudio 打开, 或 msprof --export=on --output=$PROFILING_DIR/" | tee -a "$OUTPUT_FILE"
    else
        echo "[WARN] Profiling 目录为空, ACL profiling 可能未成功采集" | tee -a "$OUTPUT_FILE"
    fi

    echo "" | tee -a "$OUTPUT_FILE"
fi

echo "============================================================" | tee -a "$OUTPUT_FILE"
echo " 测试完成!" | tee -a "$OUTPUT_FILE"
echo " 完整报告: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "============================================================" | tee -a "$OUTPUT_FILE"
