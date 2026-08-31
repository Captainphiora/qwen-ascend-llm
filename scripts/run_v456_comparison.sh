#!/bin/bash
# ============================================================
# v4 / v5 / v6 性能对比 + Profiling
#
# 用法:
#   bash scripts/run_v456_comparison.sh
#
# 每个版本执行:
#   1. Benchmark: 3 轮 greedy, 30 tokens → TPOT / Decode / TTFT
#   2. Profiling: msprof 采集 → msprof 解析为 CSV
#
# Profiling 产物 (直接看 CSV 即可):
#   profiling_<version>/PROF_xxx/mindstudio_profiler_output/
#     ├── op_statistic_xxx.csv   ← 按算子类型聚合 (最常看)
#     ├── op_summary_xxx.csv     ← 每个 kernel 详情 (含 shape/耗时/利用率)
#     ├── task_time_xxx.csv      ← 时间线
#     └── api_statistic_xxx.csv  ← ACL API 调用统计
#
# 可选: parse_profiling.py 格式化输出 (非必须, CSV 已包含全部信息)
# ============================================================

set -e

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

HF_MODEL_DIR="${HF_MODEL_DIR:-/root/models/DeepSeek-R1-Distill-Qwen-1.5B}"
DEVICE_ID="${DEVICE_ID:-0}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="results/v456_comparison_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

MSPROF=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/tools/profiler/bin/msprof

# 版本配置: label|om_path|kv_layout
declare -a VERSIONS=(
    "v4_noexpand|output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om|BSHD"
    "v5_gate_up_fuse|opt_models/v5_gate_up_fuse_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v5_gate_up_fuse_310b.om|BSHD"
    "v6_transpose_elim|opt_models/v6_transpose_elim_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v6_transpose_elim_310b.om|BHSD"
)

echo "============================================================"
echo " v4 / v5 / v6 性能对比 + Profiling"
echo " 时间: $TIMESTAMP"
echo " 输出: $RESULT_DIR/"
echo "============================================================"

for entry in "${VERSIONS[@]}"; do
    IFS='|' read -r LABEL OM_PATH KV_LAYOUT <<< "$entry"

    echo ""
    echo ">>> [$LABEL] OM=$OM_PATH  KV=$KV_LAYOUT"

    if [ ! -f "$OM_PATH" ]; then
        echo "    [SKIP] OM 不存在"
        continue
    fi

    # Benchmark
    echo "    Benchmark..."
    python benchmarks/benchmark.py \
        --om_model_path "$OM_PATH" \
        --hf_model_dir "$HF_MODEL_DIR" \
        --kv_cache_length 4096 \
        --max_prefill_length 1 \
        --max_new_tokens 30 \
        --rounds 3 --warmup 1 \
        --device_id "$DEVICE_ID" \
        --kv_cache_layout "$KV_LAYOUT" \
        --label "$LABEL" \
        2>&1 | tee "$RESULT_DIR/benchmark_${LABEL}.txt"

    # Profiling: 采集
    echo "    Profiling 采集..."
    PROF_DIR="$RESULT_DIR/profiling_${LABEL}"
    mkdir -p "$PROF_DIR"
    $MSPROF --output="$PROF_DIR" \
            --application="python benchmarks/profile_decode.py \
                --om_model_path $OM_PATH \
                --hf_model_dir $HF_MODEL_DIR \
                --kv_cache_layout $KV_LAYOUT \
                --device_id $DEVICE_ID" \
        2>&1 | tee "$RESULT_DIR/msprof_collect_${LABEL}.log"

    # Profiling: 解析为 CSV
    echo "    Profiling 解析..."
    PROF_DATA=$(find "$PROF_DIR" -maxdepth 2 -name "PROF_*" -type d | sort | tail -n 1)
    if [ -n "$PROF_DATA" ]; then
        $MSPROF --export=on --output="$PROF_DATA" --type=text --summary-format=csv \
            2>&1 | tee "$RESULT_DIR/msprof_export_${LABEL}.log"
        echo "    CSV 输出: $PROF_DATA/mindstudio_profiler_output/"

        # 可选: 格式化打印
        if [ -f "benchmarks/parse_profiling.py" ]; then
            python benchmarks/parse_profiling.py --prof_dir "$PROF_DATA" --label "$LABEL" \
                2>&1 | tee "$RESULT_DIR/analysis_${LABEL}.txt"
        fi
    fi

    echo "    [$LABEL] 完成"
done

# 汇总
echo ""
echo "============================================================"
echo " 汇总"
echo "============================================================"
for f in "$RESULT_DIR"/benchmark_*.txt; do
    [ -f "$f" ] || continue
    label=$(basename "$f" .txt | sed 's/benchmark_//')
    tpot=$(grep "TPOT" "$f" | grep -oP '[\d.]+(?= ms)' | head -1)
    decode=$(grep "Decode 速度" "$f" | grep -oP '[\d.]+(?= tokens)' | head -1)
    ttft=$(grep "TTFT" "$f" | grep -oP '[\d.]+(?= ms)' | head -1)
    printf "  %-25s TPOT=%s ms  Decode=%s tok/s  TTFT=%s ms\n" "$label" "$tpot" "$decode" "$ttft"
done
echo ""
echo "结果目录: $RESULT_DIR/"
echo "Profiling CSV: $RESULT_DIR/profiling_*/PROF_*/mindstudio_profiler_output/*.csv"
echo "============================================================"
