#!/bin/bash
# ============================================================
# 通用 Profiling 采集 & 导出脚本
#
# 用法:
#   bash profiling/run_profiling.sh                          # 使用默认配置
#   bash profiling/run_profiling.sh --script=inference.py    # 指定目标脚本
#   bash profiling/run_profiling.sh --script="inference.py --model qwen2"
#   bash profiling/run_profiling.sh --output=./my_prof_out
#   bash profiling/run_profiling.sh --analyze-only=./profiling_output/PROF_xxx
#   bash profiling/run_profiling.sh --no-analyze             # 仅采集,不分析
#
# 输出:
#   1. msprof 原始数据 (PROF_* 目录)
#   2. 导出的 CSV / Timeline JSON
#   3. 文本分析报告 (profiling_report_<timestamp>.txt)
# ============================================================

set -e

# ============================================================
# 【参数配置区】在这里修改, 直接运行即可 (也可通过命令行覆盖)
# ============================================================

# ---------- 要 profiling 的目标脚本 ----------
# 可以是任意 Python 脚本, 也可以是 shell 脚本 (如 profiling/run_inference.sh)
# 当 SCRIPT 为 .sh 文件时, 用 bash 执行; 为 .py 文件时, 用 python3 执行
SCRIPT="profiling/run_inference.sh"
SCRIPT_ARGS=""

# ---------- Profiling 输出目录 ----------
OUTPUT_DIR="./profiling_output"

# ---------- CANN 安装路径 ----------
CANN_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}"

# ---------- msprof 采集选项 ----------
TASK_TIME="on"
AI_CORE="on"
AIC_METRICS="PipeUtilization"
AIC_MODE="task-based"
AICPU="on"
RUNTIME_API="on"
HCCL="off"
MSPROFTX="off"
L2="off"

# ---------- 导出选项: summary / timeline / all ----------
EXPORT_TYPE="all"

# ---------- 控制标志 ----------
RUN_COLLECT=true
RUN_ANALYZE=true
ANALYZE_ONLY_DIR=""

# ============================================================
# 解析命令行参数
# ============================================================
for arg in "$@"; do
    case "$arg" in
        --script=*)       
            FULL="${arg#*=}"
            SCRIPT=$(echo "$FULL" | awk '{print $1}')
            SCRIPT_ARGS=$(echo "$FULL" | cut -s -d' ' -f2-)
            ;;
        --output=*)       OUTPUT_DIR="${arg#*=}" ;;
        --cann-home=*)    CANN_HOME="${arg#*=}" ;;
        --task-time=*)    TASK_TIME="${arg#*=}" ;;
        --ai-core=*)      AI_CORE="${arg#*=}" ;;
        --aic-metrics=*)  AIC_METRICS="${arg#*=}" ;;
        --aic-mode=*)     AIC_MODE="${arg#*=}" ;;
        --aicpu=*)        AICPU="${arg#*=}" ;;
        --runtime-api=*)  RUNTIME_API="${arg#*=}" ;;
        --hccl=*)         HCCL="${arg#*=}" ;;
        --l2=*)           L2="${arg#*=}" ;;
        --export=*)       EXPORT_TYPE="${arg#*=}" ;;
        --no-analyze)     RUN_ANALYZE=false ;;
        --analyze-only=*) ANALYZE_ONLY_DIR="${arg#*=}"; RUN_COLLECT=false ;;
        --help|-h)
            sed -n '2,14p' "$0"
            echo ""
            echo "OPTIONS:"
            echo "  --script=<path [args]>   目标 Python 脚本及其参数"
            echo "  --output=<dir>           Profiling 输出目录 (默认 ./profiling_output)"
            echo "  --cann-home=<path>       CANN 安装路径"
            echo "  --ai-core=on|off         AI Core 采集开关"
            echo "  --aic-metrics=<metric>   PipeUtilization|ArithmeticUtilization|Memory|..."
            echo "  --aicpu=on|off           AI CPU 算子采集开关"
            echo "  --runtime-api=on|off     Host Runtime API 采集开关"
            echo "  --hccl=on|off            集合通信采集开关"
            echo "  --l2=on|off              L2 Cache 采集开关"
            echo "  --export=summary|timeline|all  导出类型"
            echo "  --no-analyze             仅采集和导出,不运行分析"
            echo "  --analyze-only=<PROF_DIR> 跳过采集,仅对已有数据做分析"
            exit 0
            ;;
    esac
done

# ============================================================
# 环境检查
# ============================================================
if [ ! -d "$CANN_HOME" ]; then
    echo "[ERROR] 未找到 CANN: $CANN_HOME"
    echo "        请设置 --cann-home 或 ASCEND_TOOLKIT_HOME 环境变量"
    exit 1
fi

source "${CANN_HOME}/set_env.sh" 2>/dev/null || true

TOOL_DIR="${CANN_HOME}/tools/profiler/profiler_tool/analysis"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# 第一步: 采集 Profiling 数据
# ============================================================
if [ "$RUN_COLLECT" = true ]; then
    if [ ! -f "$SCRIPT" ]; then
        echo "[ERROR] 脚本不存在: $SCRIPT"
        exit 1
    fi

    # 根据文件后缀决定执行方式
    case "$SCRIPT" in
        *.sh)  EXEC_CMD="bash $SCRIPT $SCRIPT_ARGS" ;;
        *.py)  EXEC_CMD="python3 $SCRIPT $SCRIPT_ARGS" ;;
        *)     EXEC_CMD="python3 $SCRIPT $SCRIPT_ARGS" ;;
    esac

    echo ""
    echo "============================================================"
    echo " [1/3] 采集 Profiling 数据"
    echo "============================================================"
    echo "[INFO] 目标: $EXEC_CMD"
    echo "[INFO] 输出: $OUTPUT_DIR"
    echo "[INFO] 配置: ai-core=$AI_CORE, metrics=$AIC_METRICS, mode=$AIC_MODE"
    echo "[INFO]        aicpu=$AICPU, runtime-api=$RUNTIME_API, hccl=$HCCL, l2=$L2"
    echo ""

    mkdir -p "$OUTPUT_DIR"

    msprof \
        --output="$OUTPUT_DIR" \
        --task-time="$TASK_TIME" \
        --ai-core="$AI_CORE" \
        --aic-metrics="$AIC_METRICS" \
        --aic-mode="$AIC_MODE" \
        --aicpu="$AICPU" \
        --runtime-api="$RUNTIME_API" \
        --hccl="$HCCL" \
        --msproftx="$MSPROFTX" \
        --l2="$L2" \
        $EXEC_CMD

    echo ""
    echo "[INFO] 采集完成"
fi

# ============================================================
# 第二步: 定位 PROF 目录并导出
# ============================================================
if [ -n "$ANALYZE_ONLY_DIR" ]; then
    PROF_DIR="$(realpath "$ANALYZE_ONLY_DIR")"
else
    PROF_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "PROF_*" | sort | tail -1)
fi

if [ -z "$PROF_DIR" ] || [ ! -d "$PROF_DIR" ]; then
    echo "[ERROR] 未找到 PROF 目录"
    exit 1
fi

PROF_DIR=$(realpath "$PROF_DIR")
echo "[INFO] PROF 目录: $PROF_DIR"

echo ""
echo "============================================================"
echo " [2/3] 导出分析数据"
echo "============================================================"

run_msprof_export() {
    local type="$1"
    echo "[INFO] 导出 ${type} ..."
    if [ -d "$TOOL_DIR" ]; then
        PYTHONPATH="$TOOL_DIR" python3 -c "
import sys
sys.argv = ['msprof', 'export', '${type}', '-dir', '${PROF_DIR}']
from msinterface.msprof_entrance import MsprofEntrance
MsprofEntrance().main()
" || echo "[WARN] ${type} 导出可能不完整"
    else
        msprof --export=on --output="$PROF_DIR" || echo "[WARN] msprof export 可能不完整"
    fi
}

case "$EXPORT_TYPE" in
    summary)  run_msprof_export summary ;;
    timeline) run_msprof_export timeline ;;
    all)
        run_msprof_export summary
        run_msprof_export timeline
        ;;
esac

echo "[INFO] 导出完成"

# ============================================================
# 第三步: 运行 Python 分析
# ============================================================
if [ "$RUN_ANALYZE" = true ]; then
    echo ""
    echo "============================================================"
    echo " [3/3] 分析 Profiling 数据"
    echo "============================================================"

    REPORT_FILE="${OUTPUT_DIR}/profiling_report_${TIMESTAMP}.txt"

    python3 "${SCRIPT_DIR}/analyze.py" \
        --prof-dir "$PROF_DIR" \
        --output "$REPORT_FILE"

    echo ""
    echo "[INFO] 分析报告: $REPORT_FILE"
fi

echo ""
echo "============================================================"
echo " 完成!"
echo " PROF 数据:   $PROF_DIR"
echo " Timeline:    可用 chrome://tracing 或 https://ui.perfetto.dev 可视化"
if [ "$RUN_ANALYZE" = true ]; then
    echo " 分析报告:   $REPORT_FILE"
fi
echo "============================================================"
