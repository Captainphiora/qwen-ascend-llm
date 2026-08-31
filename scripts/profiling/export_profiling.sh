#!/bin/bash

# 昇腾 CANN 9.0 Profiling 数据导出脚本
# 用法: bash export_profiling.sh <PROF目录路径> [summary|timeline|all]
# 示例: bash export_profiling.sh ./profiling_sampling_data/PROF_000001_20260814030921365_01179043FLPGNBPE all

set -e

PROF_DIR="${1:?请指定 PROF 目录路径，例如: ./profiling_sampling_data/PROF_000001_xxx}"
EXPORT_TYPE="${2:-all}"

PROF_DIR=$(realpath "$PROF_DIR")

CANN_HOME="/usr/local/Ascend/cann-9.0.0"
TOOL_DIR="${CANN_HOME}/tools/profiler/profiler_tool/analysis"

if [ ! -d "$CANN_HOME" ]; then
    echo "[ERROR] 未找到 CANN 目录: $CANN_HOME"
    exit 1
fi

if [ ! -d "$PROF_DIR" ]; then
    echo "[ERROR] PROF 目录不存在: $PROF_DIR"
    exit 1
fi

source "${CANN_HOME}/set_env.sh"

run_msprof_export() {
    local type="$1"
    echo "[INFO] 正在导出 ${type} ..."
    cd "$TOOL_DIR" && python3 -c "
from msinterface.msprof_entrance import MsprofEntrance
import sys
sys.argv = ['msprof', 'export', '${type}', '-dir', '${PROF_DIR}']
MsprofEntrance().main()
"
    echo "[INFO] ${type} 导出完成"
}

case "$EXPORT_TYPE" in
    summary)
        run_msprof_export summary
        ;;
    timeline)
        run_msprof_export timeline
        ;;
    all)
        run_msprof_export summary
        run_msprof_export timeline
        ;;
    *)
        echo "[ERROR] 不支持的导出类型: $EXPORT_TYPE (可选: summary, timeline, all)"
        exit 1
        ;;
esac

echo ""
echo "[DONE] 结果输出目录: ${PROF_DIR}/mindstudio_profiler_output/"
ls "${PROF_DIR}/mindstudio_profiler_output/"
