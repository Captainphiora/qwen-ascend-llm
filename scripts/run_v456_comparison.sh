#!/bin/bash
# ============================================================
# v4 / v5 / v6 完整性能对比 + Profiling
#
# 用法:
#   bash scripts/run_v456_comparison.sh
#
# 流程 (每个版本):
#   1. Benchmark (3 轮, greedy, 30 tokens)
#   2. Profiling (msprof 采集 → 解析 → 分析)
#
# 输出:
#   results/v456_comparison_<timestamp>/
#     ├── benchmark_v4.txt
#     ├── benchmark_v5.txt
#     ├── benchmark_v6.txt
#     ├── profiling_v4/
#     ├── profiling_v5/
#     └── profiling_v6/
# ============================================================

set -e

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

HF_MODEL_DIR="${HF_MODEL_DIR:-/root/models/DeepSeek-R1-Distill-Qwen-1.5B}"
DEVICE_ID="${DEVICE_ID:-0}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="results/v456_comparison_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# CANN 环境
MSPROF=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/tools/profiler/bin/msprof

# ============================================================
# 版本配置: (label, om_path, kv_layout)
# ============================================================
declare -a VERSIONS=(
    "v4_noexpand|output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om|BSHD"
    "v5_gate_up_fuse|opt_models/v5_gate_up_fuse_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v5_gate_up_fuse_310b.om|BSHD"
    "v6_transpose_elim|opt_models/v6_transpose_elim_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v6_transpose_elim_310b.om|BHSD"
)

echo "============================================================"
echo " v4 / v5 / v6 完整性能对比"
echo "============================================================"
echo " 时间: $TIMESTAMP"
echo " 设备: Device $DEVICE_ID"
echo " 输出: $RESULT_DIR/"
echo "============================================================"
echo ""

for entry in "${VERSIONS[@]}"; do
    IFS='|' read -r LABEL OM_PATH KV_LAYOUT <<< "$entry"

    echo ""
    echo "============================================================"
    echo " [$LABEL] 开始测试"
    echo " OM: $OM_PATH"
    echo " KV Layout: $KV_LAYOUT"
    echo "============================================================"

    if [ ! -f "$OM_PATH" ]; then
        echo "[WARN] OM 文件不存在: $OM_PATH, 跳过"
        continue
    fi

    # --- Benchmark ---
    echo ""
    echo ">>> [$LABEL] Benchmark (3 轮, 30 tokens)..."
    BENCH_FILE="$RESULT_DIR/benchmark_${LABEL}.txt"
    python benchmarks/benchmark.py \
        --om_model_path "$OM_PATH" \
        --hf_model_dir "$HF_MODEL_DIR" \
        --kv_cache_length 4096 \
        --max_prefill_length 1 \
        --max_new_tokens 30 \
        --rounds 3 \
        --warmup 1 \
        --device_id "$DEVICE_ID" \
        --kv_cache_layout "$KV_LAYOUT" \
        --label "$LABEL" \
        2>&1 | tee "$BENCH_FILE"
    echo "[$LABEL] Benchmark 完成: $BENCH_FILE"

    # --- Profiling ---
    echo ""
    echo ">>> [$LABEL] Profiling 采集..."
    PROF_DIR="$RESULT_DIR/profiling_${LABEL}"
    mkdir -p "$PROF_DIR"

    $MSPROF --output="$PROF_DIR" \
            --aic-metrics=PipeUtilization \
            --application="python benchmarks/profile_decode.py \
                --om_model_path $OM_PATH \
                --hf_model_dir $HF_MODEL_DIR \
                --kv_cache_layout $KV_LAYOUT \
                --device_id $DEVICE_ID \
                --max_new_tokens 20" \
        2>&1 | tee "$RESULT_DIR/profiling_collect_${LABEL}.log"

    # 解析 CSV
    echo ">>> [$LABEL] Profiling 解析..."
    PROF_DATA=$(find "$PROF_DIR" -maxdepth 2 -name "PROF_*" -type d | sort | tail -n 1)
    if [ -n "$PROF_DATA" ]; then
        $MSPROF --export=on --output="$PROF_DATA" --type=text --summary-format=csv \
            2>&1 | tee -a "$RESULT_DIR/profiling_parse_${LABEL}.log"

        # 分析
        if [ -f "benchmarks/parse_profiling.py" ]; then
            python benchmarks/parse_profiling.py \
                --prof_dir "$PROF_DATA" \
                --label "$LABEL" \
                2>&1 | tee "$RESULT_DIR/profiling_analysis_${LABEL}.txt"
        fi
    else
        echo "[WARN] 未找到 PROF_* 目录"
    fi

    echo "[$LABEL] 全部完成"
    echo ""
done

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================================"
echo " 全部测试完成!"
echo "============================================================"
echo " 结果目录: $RESULT_DIR/"
echo ""
echo " Benchmark 文件:"
ls -1 "$RESULT_DIR"/benchmark_*.txt 2>/dev/null | while read f; do echo "   $f"; done
echo ""
echo " Profiling 目录:"
ls -d "$RESULT_DIR"/profiling_v* 2>/dev/null | while read f; do echo "   $f"; done
echo ""

# 提取关键指标汇总
echo " 关键指标汇总:"
echo " -----------------------------------------------"
for f in "$RESULT_DIR"/benchmark_*.txt; do
    label=$(basename "$f" .txt | sed 's/benchmark_//')
    tpot=$(grep "TPOT" "$f" | grep -oP '[\d.]+(?= ms)' | head -1)
    decode=$(grep "Decode 速度" "$f" | grep -oP '[\d.]+(?= tokens)' | head -1)
    ttft=$(grep "TTFT" "$f" | grep -oP '[\d.]+(?= ms)' | head -1)
    printf "   %-25s TPOT=%s ms  Decode=%s tok/s  TTFT=%s ms\n" "$label" "$tpot" "$decode" "$ttft"
done
echo " -----------------------------------------------"
echo "============================================================"
