#!/bin/bash
# ============================================================
# v4 / v5 / v6 性能对比 + Profiling
#
# 用法:
#   bash scripts/run_v456_comparison.sh                 # 默认含格式化分析
#   bash scripts/run_v456_comparison.sh --no-analysis   # 只采集+解析CSV,不跑分析脚本
#
# 产物结构:
#   results/v456_<timestamp>/
#     ├── benchmark_v4_noexpand.txt          ← 推理性能
#     ├── benchmark_v5_gate_up_fuse.txt
#     ├── benchmark_v6_transpose_elim.txt
#     ├── analysis_v4_noexpand.txt           ← 算子级分析 (--no-analysis 时不生成)
#     ├── analysis_v5_gate_up_fuse.txt
#     ├── analysis_v6_transpose_elim.txt
#     ├── report_v4_noexpand.txt             ← 完整报告 (含数据搬运, 同 profiling/analyze.py)
#     ├── report_v5_gate_up_fuse.txt
#     ├── report_v6_transpose_elim.txt
#     ├── profiling_v4_noexpand/             ← msprof 原始数据 + CSV
#     │   └── PROF_xxx/mindstudio_profiler_output/
#     │       ├── op_statistic_xxx.csv       ← 按算子类型聚合
#     │       ├── op_summary_xxx.csv         ← 每个 kernel 详情
#     │       └── ...
#     ├── profiling_v5_gate_up_fuse/
#     └── profiling_v6_transpose_elim/
# ============================================================

set -e

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

HF_MODEL_DIR="${HF_MODEL_DIR:-../models/DeepSeek-R1-Distill-Qwen-1.5B}"
DEVICE_ID="${DEVICE_ID:-0}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="results/v456_${TIMESTAMP}"
RUN_ANALYSIS=true

# 解析参数
for arg in "$@"; do
    case $arg in
        --no-analysis) RUN_ANALYSIS=false ;;
    esac
done

mkdir -p "$RESULT_DIR"

MSPROF=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/tools/profiler/bin/msprof
MSPROF_ANALYZE="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/tools/profiler/profiler_tool/analysis/msprof/msprof.py"

# 版本配置: label|om_path|kv_layout
declare -a VERSIONS=(
    "v4_noexpand|output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om|BSHD"
    "v5_gate_up_fuse|opt_models/v5_gate_up_fuse_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v5_gate_up_fuse_310b.om|BSHD"
    "v6_transpose_elim|opt_models/v6_transpose_elim_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v6_transpose_elim_310b.om|BHSD"
)

CLEANUP_SCRIPT="$PROJECT_DIR/scripts/cleanup_memory.sh"

echo "============================================================"
echo " v4 / v5 / v6 性能对比 + Profiling"
echo " 时间:   $TIMESTAMP"
echo " 输出:   $RESULT_DIR/"
echo " 分析:   $RUN_ANALYSIS"
echo "============================================================"

for entry in "${VERSIONS[@]}"; do
    IFS='|' read -r LABEL OM_PATH KV_LAYOUT <<< "$entry"

    echo ""
    echo ">>> [$LABEL] OM=$OM_PATH  KV=$KV_LAYOUT"

    if [ ! -f "$OM_PATH" ]; then
        echo "    [SKIP] OM 不存在"
        continue
    fi

    # --- 清理内存 ---
    echo "    [0/4] 清理内存..."
    if [ -f "$CLEANUP_SCRIPT" ]; then
        sudo bash "$CLEANUP_SCRIPT" 2>&1 | tail -5
    else
        echo "    [WARN] cleanup_memory.sh 不存在, 跳过"
    fi

    # --- Benchmark ---
    echo "    [1/4] Benchmark..."
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

    # --- 清理内存 (benchmark 后、profiling 前) ---
    echo "    [2/4] 清理内存 (profiling 前)..."
    if [ -f "$CLEANUP_SCRIPT" ]; then
        sudo bash "$CLEANUP_SCRIPT" 2>&1 | tail -3
    fi

    # --- Profiling: msprof 采集 ---
    echo "    [3/4] Profiling 采集..."
    PROF_DIR="$(realpath "$RESULT_DIR")/profiling_${LABEL}"
    mkdir -p "$PROF_DIR"
    $MSPROF --output="$PROF_DIR" \
            --runtime-api=on \
            --application="python $(realpath benchmarks/profile_decode.py) \
                --om_model_path $(realpath $OM_PATH) \
                --hf_model_dir $HF_MODEL_DIR \
                --kv_cache_layout $KV_LAYOUT \
                --device_id $DEVICE_ID" \
        2>&1 | tee "$RESULT_DIR/msprof_collect_${LABEL}.log"

    # --- Profiling: msprof 解析为 CSV ---
    echo "    [4/4] Profiling 解析..."
    PROF_DATA=$(find "$PROF_DIR" -maxdepth 2 -name "PROF_*" -type d | sort | tail -n 1)
    if [ -n "$PROF_DATA" ]; then
        PROF_DATA=$(realpath "$PROF_DATA")
        # 兼容不同 CANN 版本: 优先 --export=on, 失败则用 --parse=on
        if ! $MSPROF --output="$PROF_DATA" --export=on --type=text --summary-format=csv 2>&1 | tee "$RESULT_DIR/msprof_export_${LABEL}.log"; then
            echo "    [WARN] --export=on 失败, 尝试 --parse=on"
            $MSPROF --output="$PROF_DATA" --parse=on --type=text --summary-format=csv \
                2>&1 | tee "$RESULT_DIR/msprof_export_${LABEL}.log"
        fi

        # 格式化分析 (可通过 --no-analysis 跳过)
        if [ "$RUN_ANALYSIS" = true ] && [ -f "benchmarks/parse_profiling.py" ]; then
            python benchmarks/parse_profiling.py \
                --prof_dir "$PROF_DATA" --label "$LABEL" \
                2>&1 | tee "$RESULT_DIR/analysis_${LABEL}.txt"
        fi

        if [ "$RUN_ANALYSIS" = true ] && [ -f "profiling/analyze.py" ]; then
            python profiling/analyze.py \
                --prof-dir "$PROF_DATA" \
                --output "$RESULT_DIR/report_${LABEL}.txt" \
                2>&1 | tail -5
        fi
    fi

    echo "    [$LABEL] done"
done

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================================"
echo " 汇总"
echo "============================================================"
printf "  %-25s %10s %14s %10s\n" "版本" "TPOT(ms)" "Decode(tok/s)" "TTFT(ms)"
echo "  -----------------------------------------------------------"
for f in "$RESULT_DIR"/benchmark_*.txt; do
    [ -f "$f" ] || continue
    label=$(basename "$f" .txt | sed 's/benchmark_//')
    tpot=$(grep "TPOT" "$f" | grep -oP '[\d.]+(?= ms)' | head -1)
    decode=$(grep "Decode 速度" "$f" | grep -oP '[\d.]+(?= tokens)' | head -1)
    ttft=$(grep "TTFT" "$f" | grep -oP '[\d.]+(?= ms)' | head -1)
    printf "  %-25s %10s %14s %10s\n" "$label" "$tpot" "$decode" "$ttft"
done
echo ""
echo " 结果目录: $RESULT_DIR/"
echo " CSV 路径: $RESULT_DIR/profiling_*/PROF_*/mindstudio_profiler_output/*.csv"
[ "$RUN_ANALYSIS" = true ] && echo " 算子分析:   $RESULT_DIR/analysis_*.txt"
[ "$RUN_ANALYSIS" = true ] && echo " 完整报告:   $RESULT_DIR/report_*.txt (含数据搬运)"
echo "============================================================"
