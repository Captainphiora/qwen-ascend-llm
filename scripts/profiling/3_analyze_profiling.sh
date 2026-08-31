#!/bin/bash
# ============================================================
# 脚本3: Profiling 数据分析
# 解析 msprof 采集的原始数据，输出算子统计、API 统计、数据搬运统计
#
# 使用方式:
#   bash scripts/3_analyze_profiling.sh --input=./profiling_data/<timestamp>
#   bash scripts/3_analyze_profiling.sh --input=./profiling_data/20260731_201237
#
# 示例:
#   # 分析最近一次采集的 profiling 数据
#   bash scripts/3_analyze_profiling.sh --input=$(ls -td ./profiling_data/*/ | head -1)
#
#   # 分析指定目录
#   bash scripts/3_analyze_profiling.sh --input=./profiling_data/20260803_163000
#
# 输出:
#   解析后的 CSV 文件在 <input>/PROF_*/mindstudio_profiler_output/ 下
#   日志: scripts/logs/3_analyze_profiling_<timestamp>.log
#
# 生成的 CSV 文件:
#   op_statistic_*.csv   - Device 算子类型级统计（按耗时排序）
#   op_summary_*.csv     - Device 每个算子实例详情
#   api_statistic_*.csv  - Host ACL API 调用统计（含 Memcpy）
#   task_time_*.csv      - Device 任务时间线
# ============================================================

set -e
source ~/.bashrc_cann900

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# 解析参数
INPUT_DIR=""
for arg in "$@"; do
    case "$arg" in
        --input=*) INPUT_DIR="${arg#*=}" ;;
        --help|-h)
            sed -n '2,28p' "$0"
            exit 0
            ;;
    esac
done

if [ -z "$INPUT_DIR" ] || [ ! -d "$INPUT_DIR" ]; then
    echo "[ERROR] 请指定有效的 profiling 数据目录: --input=<dir>"
    echo "可用目录:"
    ls -td ./profiling_data/*/ 2>/dev/null | head -5
    exit 1
fi

MSPROF="${ASCEND_TOOLKIT_HOME}/tools/profiler/bin/msprof"
if [ ! -f "$MSPROF" ]; then
    echo "[ERROR] msprof 未找到: $MSPROF"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="scripts/logs/3_analyze_profiling_${TIMESTAMP}.log"
mkdir -p scripts/logs

echo "============================================================" | tee "$LOG_FILE"
echo " [3] Profiling 数据分析" | tee -a "$LOG_FILE"
echo " 输入目录: ${INPUT_DIR}" | tee -a "$LOG_FILE"
echo " msprof: ${MSPROF}" | tee -a "$LOG_FILE"
echo " Time: ${TIMESTAMP}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 找到 PROF_* 子目录
PROF_SUBDIR=$(find "$INPUT_DIR" -maxdepth 1 -type d -name "PROF_*" | sort | tail -1)
if [ -z "$PROF_SUBDIR" ]; then
    echo "[ERROR] 未找到 PROF_* 子目录" | tee -a "$LOG_FILE"
    find "$INPUT_DIR" -maxdepth 2 -type d | tee -a "$LOG_FILE"
    exit 1
fi

echo ">>> PROF 目录: $PROF_SUBDIR" | tee -a "$LOG_FILE"
echo ">>> 目录结构:" | tee -a "$LOG_FILE"
ls -la "$PROF_SUBDIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 解析
echo ">>> 执行 msprof --export=on ..." | tee -a "$LOG_FILE"
"$MSPROF" --export=on --output="$PROF_SUBDIR" 2>&1 | tee -a "$LOG_FILE" || true
echo "" | tee -a "$LOG_FILE"

# 展示结果
OUTPUT_DIR=$(find "$PROF_SUBDIR" -type d -name "mindstudio_profiler_output" | head -1)
if [ -z "$OUTPUT_DIR" ]; then
    echo "[ERROR] 解析后未生成 mindstudio_profiler_output 目录" | tee -a "$LOG_FILE"
    exit 1
fi

echo ">>> 解析完成，CSV 文件:" | tee -a "$LOG_FILE"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Device 算子统计
OP_STAT=$(find "$OUTPUT_DIR" -name "op_statistic*.csv" | sort | tail -1)
if [ -n "$OP_STAT" ]; then
    echo "┌── Device 算子耗时统计 ──────────────────────────────────────────┐" | tee -a "$LOG_FILE"
    echo ">>> 文件: $OP_STAT" | tee -a "$LOG_FILE"
    head -21 "$OP_STAT" | column -t -s',' 2>/dev/null | tee -a "$LOG_FILE" || head -21 "$OP_STAT" | tee -a "$LOG_FILE"
    echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# Host API 统计
API_STAT=$(find "$OUTPUT_DIR" -name "api_statistic*.csv" | sort | tail -1)
if [ -n "$API_STAT" ]; then
    echo "┌── Host ACL API 耗时统计 ────────────────────────────────────────┐" | tee -a "$LOG_FILE"
    echo ">>> 文件: $API_STAT" | tee -a "$LOG_FILE"
    head -21 "$API_STAT" | column -t -s',' 2>/dev/null | tee -a "$LOG_FILE" || head -21 "$API_STAT" | tee -a "$LOG_FILE"
    echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # 数据搬运统计
    echo "┌── 数据搬运 (H2D/D2H) 耗时统计 ─────────────────────────────────┐" | tee -a "$LOG_FILE"
    head -1 "$API_STAT" | column -t -s',' 2>/dev/null | tee -a "$LOG_FILE"
    grep -i "memcpy\|MemCopy\|InputCopy\|OutputCopy\|Copy" "$API_STAT" | column -t -s',' 2>/dev/null | tee -a "$LOG_FILE" || true
    echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 分析完成!" | tee -a "$LOG_FILE"
echo " CSV 结果: ${OUTPUT_DIR}/" | tee -a "$LOG_FILE"
echo " 日志: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo " 可视化: 用 MindStudio 打开 ${PROF_SUBDIR}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
