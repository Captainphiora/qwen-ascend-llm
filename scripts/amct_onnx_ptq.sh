#!/bin/bash
# ============================================================
# AMCT ONNX PTQ 量化 + ATC 编译脚本
#
# 用法:
#   bash scripts/amct_onnx_ptq.sh [--npu_id=3] [--num_samples=8]
# ============================================================

set -e
source ~/.bashrc_cann900
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen_ascend_cann900

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 默认配置 ----
NPU_ID="3"
NUM_SAMPLES=8
KV_CACHE_LENGTH=4096
MODEL_PATH="output/onnx_changed_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx"
HF_MODEL_DIR="/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
CALIB_FILE="/usr/local/Ascend/atb-models/examples/convert/model_slim/boolq.jsonl"
OUTPUT_DIR="output/amct_onnx_ptq"
OM_OUTPUT="output/om_ptq_910/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_ptq"
# ---- 配置结束 ----

for arg in "$@"; do
    case "$arg" in
        --npu_id=*) NPU_ID="${arg#*=}" ;;
        --num_samples=*) NUM_SAMPLES="${arg#*=}" ;;
        --kv_cache_length=*) KV_CACHE_LENGTH="${arg#*=}" ;;
        --model_path=*) MODEL_PATH="${arg#*=}" ;;
        --hf_model_dir=*) HF_MODEL_DIR="${arg#*=}" ;;
    esac
done

export ASCEND_RT_VISIBLE_DEVICES="$NPU_ID"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="scripts/logs/amct_onnx_ptq_${TIMESTAMP}.log"
mkdir -p scripts/logs "$OUTPUT_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo " AMCT ONNX PTQ 量化" | tee -a "$LOG_FILE"
echo " NPU: $NPU_ID  Samples: $NUM_SAMPLES  KV: $KV_CACHE_LENGTH" | tee -a "$LOG_FILE"
echo " Model: $MODEL_PATH" | tee -a "$LOG_FILE"
echo " Time: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Phase 1: 校准量化
echo ">>> [Phase 1] AMCT 校准量化..." | tee -a "$LOG_FILE"
python3 scripts/amct_onnx_calibrate.py \
    --model_path "$MODEL_PATH" \
    --hf_model_dir "$HF_MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --calib_file "$CALIB_FILE" \
    --num_samples "$NUM_SAMPLES" \
    --kv_cache_length "$KV_CACHE_LENGTH" \
    2>&1 | tee -a "$LOG_FILE"

# Phase 2: ATC 编译 OM
DEPLOY_ONNX="${OUTPUT_DIR}/model_deploy_deploy_model.onnx"
if [ -f "$DEPLOY_ONNX" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo ">>> [Phase 2] ATC 编译 OM..." | tee -a "$LOG_FILE"
    python3 export/onnx2om.py \
        --hf_model_dir="$HF_MODEL_DIR" \
        --onnx_model_path="$DEPLOY_ONNX" \
        --om_model_path="$OM_OUTPUT" \
        --kv_cache_length="$KV_CACHE_LENGTH" \
        --max_prefill_length=1 \
        --max_batch=1 \
        --cpu_thread=16 \
        --soc_version=auto \
        2>&1 | tee -a "$LOG_FILE"
    echo ">>> OM 编译完成: ${OM_OUTPUT}.om" | tee -a "$LOG_FILE"
else
    echo "[ERROR] Deploy ONNX 不存在: $DEPLOY_ONNX" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 完成! 日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
