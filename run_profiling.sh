#!/bin/bash
# ============================================================
# Profiling 采集与解析脚本
#
# 用法:
#   第一步 - 采集:  python profile_inference.py
#   第二步 - 解析:  bash run_profiling.sh --parse
# ============================================================

set -e

# CANN 环境配置
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0_backup
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/aarch64-linux/lib64:$ASCEND_HOME_PATH/aarch64-linux/devlib:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export PATH=$ASCEND_HOME_PATH/tools/profiler/bin:$PATH
MSPROF=$ASCEND_HOME_PATH/tools/profiler/bin/msprof

# 工作目录
WORK_DIR=$(cd "$(dirname "$0")" && pwd)
PROFILING_DATA=$WORK_DIR/profiling_data

# ============================================================
# 解析模式
# ============================================================
if [ "$1" == "--parse" ]; then
    PROF_DIRS=($(find $PROFILING_DATA -maxdepth 2 -name "PROF_*" -type d | sort))
    if [ ${#PROF_DIRS[@]} -eq 0 ]; then
        echo "[ERROR] 找不到 PROF_* 目录"
        echo "[ERROR] 请先运行采集: python profile_inference.py"
        exit 1
    fi

    if [ ${#PROF_DIRS[@]} -eq 1 ]; then
        PROF_DIR=${PROF_DIRS[0]}
    else
        echo "[INFO] 发现多个 PROF_* 目录，请选择要解析的目录:"
        for i in "${!PROF_DIRS[@]}"; do
            echo "  [$((i+1))] ${PROF_DIRS[$i]}"
        done
        echo ""
        read -p "请输入编号 [1-${#PROF_DIRS[@]}]: " choice
        if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#PROF_DIRS[@]} ]; then
            echo "[ERROR] 无效的选择: $choice"
            exit 1
        fi
        PROF_DIR=${PROF_DIRS[$((choice-1))]}
    fi

    echo "[INFO] 找到 profiling 数据: $PROF_DIR"
    echo "[INFO] 开始解析..."

    $MSPROF --export=on \
            --output=$PROF_DIR \
            --type=text \
            --summary-format=csv

    echo ""
    echo "[INFO] 解析完成!"
    echo "[INFO] 生成的文件:"
    find $PROF_DIR -name "*.csv" 2>/dev/null | sort | while read f; do
        echo "  $f"
    done
    echo ""
    echo "[INFO] 运行分析: python analyze_profiling.py"
    exit 0
fi

echo "用法:"
echo "  采集: python profile_inference.py"
echo "  解析: bash run_profiling.sh --parse"
echo "  分析: python analyze_profiling.py"
