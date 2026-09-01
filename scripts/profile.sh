#!/bin/bash
# ============================================================
# 通用 Profiling: msprof 采集 → 解析 CSV → 可选格式化分析
#
# 用法:
#   bash scripts/profile.sh "python my_script.py --arg1 val1"
#   bash scripts/profile.sh "python my_script.py" --label exp1
#   bash scripts/profile.sh "python my_script.py" --output-dir results/my_exp
#   bash scripts/profile.sh "python my_script.py" --no-analysis
#   bash scripts/profile.sh "python my_script.py" --msprof-args "--aic-metrics=PipeUtilization"
#
# 产物:
#   <output-dir>/
#     ├── PROF_xxx/mindstudio_profiler_output/
#     │   ├── op_statistic_xxx.csv   ← 按算子类型聚合
#     │   ├── op_summary_xxx.csv     ← 每个 kernel 详情 (shape/耗时/利用率)
#     │   ├── api_statistic_xxx.csv  ← ACL API 统计 (含数据搬运)
#     │   └── ...
#     ├── collect.log
#     ├── export.log
#     └── analysis.txt               ← 格式化分析 (可选)
# ============================================================

set -e

APPLICATION=""
LABEL=""
OUTPUT_DIR=""
RUN_ANALYSIS=true
MSPROF_EXTRA_ARGS=""

usage() {
    echo "用法: bash $0 \"<command>\" [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --label NAME         标签 (默认: profile)"
    echo "  --output-dir DIR     输出目录 (默认: profiling/<label>_<timestamp>)"
    echo "  --no-analysis        跳过格式化分析, 只保留 CSV"
    echo "  --msprof-args ARGS   传递额外参数给 msprof"
    echo ""
    echo "示例:"
    echo "  bash $0 \"python inference.py --model model.om\""
    echo "  bash $0 \"python train.py\" --label train_exp1"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --label) LABEL="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --no-analysis) RUN_ANALYSIS=false; shift ;;
        --msprof-args) MSPROF_EXTRA_ARGS="$2"; shift 2 ;;
        --help|-h) usage ;;
        *)
            if [ -z "$APPLICATION" ]; then
                APPLICATION="$1"; shift
            else
                echo "[ERROR] 多余的参数: $1"; usage
            fi
            ;;
    esac
done

[ -z "$APPLICATION" ] && { echo "[ERROR] 缺少要 profiling 的命令"; usage; }

# ============================================================
# 路径设置
# ============================================================
LABEL="${LABEL:-profile}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="profiling/${LABEL}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

MSPROF=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/tools/profiler/bin/msprof
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# msprof fork 子进程时会丢失 PYTHONPATH, 需要注入项目根目录
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "============================================================"
echo " Profiling"
echo " 命令:   $APPLICATION"
echo " 标签:   $LABEL"
echo " 输出:   $OUTPUT_DIR/"
echo " 分析:   $RUN_ANALYSIS"
echo "============================================================"

# ============================================================
# Step 1: msprof 采集
# ============================================================
echo ""
echo "[1/2] 采集..."

# msprof 新版本用 "msprof [args] <app> [app args]" 形式
# 旧版本用 --application="..." 形式
# 先尝试新语法，失败再用旧语法
if ! $MSPROF --output="$OUTPUT_DIR" $MSPROF_EXTRA_ARGS $APPLICATION \
    2>&1 | tee "$OUTPUT_DIR/collect.log" \
    || ! find "$OUTPUT_DIR" -maxdepth 2 -name "PROF_*" -type d | grep -q .; then

    echo "[INFO] 直接追加命令失败, 尝试 --application 形式..."
    $MSPROF --output="$OUTPUT_DIR" $MSPROF_EXTRA_ARGS \
            --application="$APPLICATION" \
        2>&1 | tee "$OUTPUT_DIR/collect.log"
fi

# ============================================================
# Step 2: msprof 解析为 CSV
# ============================================================
echo ""
echo "[2/2] 解析..."
PROF_DATA=$(find "$OUTPUT_DIR" -maxdepth 2 -name "PROF_*" -type d | sort | tail -n 1)

if [ -z "$PROF_DATA" ]; then
    echo "[ERROR] 未找到 PROF_* 目录, 采集可能失败"
    echo "[ERROR] 检查 $OUTPUT_DIR/collect.log"
    exit 1
fi

PROF_DATA=$(realpath "$PROF_DATA")

# 兼容不同 CANN 版本: 优先 --export=on, 失败则用 --parse=on
if ! $MSPROF --output="$PROF_DATA" --export=on --type=text --summary-format=csv \
    2>&1 | tee "$OUTPUT_DIR/export.log"; then
    echo "[WARN] --export=on 失败, 尝试 --parse=on"
    $MSPROF --output="$PROF_DATA" --parse=on --type=text --summary-format=csv \
        2>&1 | tee "$OUTPUT_DIR/export.log"
fi

# ============================================================
# 可选: 格式化分析
# ============================================================
ANALYSIS_SCRIPT="$PROJECT_DIR/benchmarks/parse_profiling.py"
if [ "$RUN_ANALYSIS" = true ] && [ -f "$ANALYSIS_SCRIPT" ]; then
    echo ""
    echo "[分析] 格式化输出..."
    python "$ANALYSIS_SCRIPT" --prof_dir "$PROF_DATA" --label "$LABEL" \
        2>&1 | tee "$OUTPUT_DIR/analysis.txt"
fi

# ============================================================
# 完成
# ============================================================
CSV_DIR=$(find "$PROF_DATA" -type d -name "mindstudio_profiler_output" | head -1)
echo ""
echo "============================================================"
echo " 完成"
echo " CSV:      ${CSV_DIR}/"
[ -f "$OUTPUT_DIR/analysis.txt" ] && echo " 分析报告: $OUTPUT_DIR/analysis.txt"
echo "============================================================"
