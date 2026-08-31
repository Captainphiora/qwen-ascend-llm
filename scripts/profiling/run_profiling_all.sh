#!/bin/bash
# ============================================================
# 一键 Profiling 脚本：采集 → 解析 → 分析
#
# 用法:
#   bash run_profiling_all.sh [OPTIONS]
#
# 选项:
#   --om_model_path     OM 模型路径 (默认: ./output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om)
#   --hf_model_dir      HuggingFace 模型目录
#   --device_id         NPU 设备 ID (默认: 0)
#   --kv_cache_length   KV Cache 长度 (默认: 4096)
#   --max_prefill_length  最大 prefill 长度 (默认: 1)
#   --max_new_tokens    生成 token 数 (默认: 20)
#   --conda_env         Conda 环境名 (默认: qwen_ascend_cann900)
# ============================================================

set -e

# 默认参数
OM_MODEL_PATH="./output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om"
HF_MODEL_DIR="/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE_ID=5
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
MAX_NEW_TOKENS=20
CONDA_ENV="qwen_ascend_cann900"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --om_model_path) OM_MODEL_PATH="$2"; shift 2 ;;
        --hf_model_dir) HF_MODEL_DIR="$2"; shift 2 ;;
        --device_id) DEVICE_ID="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --kv_cache_length) KV_CACHE_LENGTH="$2"; shift 2 ;;
        --max_prefill_length) MAX_PREFILL_LENGTH="$2"; shift 2 ;;
        --conda_env) CONDA_ENV="$2"; shift 2 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

# 生成输出目录标识（模型名 + 时间戳）
MODEL_BASENAME=$(basename "$OM_MODEL_PATH" .om)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_TAG="${MODEL_BASENAME}_${TIMESTAMP}"

WORK_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$WORK_DIR"

PROFILING_RAW_DIR="$WORK_DIR/profiling_output/${RESULT_TAG}/raw"
PROFILING_RESULT_DIR="$WORK_DIR/profiling_output/${RESULT_TAG}"
ANALYSIS_FILE="$PROFILING_RESULT_DIR/analysis_${RESULT_TAG}.txt"

mkdir -p "$PROFILING_RAW_DIR"

echo "============================================================"
echo " Profiling 一键脚本"
echo "============================================================"
echo " 模型:    $OM_MODEL_PATH"
echo " 设备:    Device $DEVICE_ID"
echo " 输出:    $PROFILING_RESULT_DIR"
echo "============================================================"
echo ""

# 激活环境
echo "[Step 0] 激活环境: $CONDA_ENV"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# 设置 CANN 环境
source ~/.bashrc_cann900
MSPROF=$ASCEND_HOME_PATH/tools/profiler/bin/msprof

echo ""

# Step 1: 采集
echo "[Step 1] 采集 Profiling 数据..."
python "$WORK_DIR/profiling_collect.py" \
    --om_model_path "$OM_MODEL_PATH" \
    --hf_model_dir "$HF_MODEL_DIR" \
    --output_dir "$PROFILING_RAW_DIR" \
    --device_id "$DEVICE_ID" \
    --kv_cache_length "$KV_CACHE_LENGTH" \
    --max_prefill_length "$MAX_PREFILL_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS"
echo ""

# Step 2: 解析（msprof 导出 CSV）
echo "[Step 2] 解析 Profiling 数据 (msprof)..."
PROF_DIR=$(find "$PROFILING_RAW_DIR" -maxdepth 2 -name "PROF_*" -type d | sort | tail -n 1)
if [ -z "$PROF_DIR" ]; then
    echo "[ERROR] 未找到 PROF_* 目录，采集可能失败"
    exit 1
fi

$MSPROF --export=on \
        --output="$PROF_DIR" \
        --type=text \
        --summary-format=csv
echo ""

# Step 3: 分析
echo "[Step 3] 分析 Profiling 数据..."
python "$WORK_DIR/profiling_analyze.py" \
    --prof_dir "$PROF_DIR" \
    --output_file "$ANALYSIS_FILE" \
    --model_name "$MODEL_BASENAME" \
    --device_id "$DEVICE_ID"
echo ""

# 完成
echo "============================================================"
echo " 完成! 输出文件:"
echo "   原始数据:  $PROFILING_RAW_DIR"
echo "   分析报告:  $ANALYSIS_FILE"
echo "============================================================"
