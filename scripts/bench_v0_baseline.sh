#!/bin/bash
# ============================================================
# v0 基线性能测试
# 使用当前未优化的 OM 模型，测量 baseline 性能
# ============================================================
set -e
cd "$(dirname "$0")/.."

VERSION="v0_baseline"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
# SUFFIX=""
SUFFIX="_sim"
OM_MODEL_PATH="./output/model_910_cann900/${MODEL_NAME}_4096_1${SUFFIX}.om"
RESULT_DIR="./benchmark_results${SUFFIX}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$RESULT_DIR"
RESULT_FILE="${RESULT_DIR}/${VERSION}_${TIMESTAMP}.txt"

DEVICE_ID=5

echo "============================================================"
echo " 性能测试: ${VERSION}"
echo " OM: ${OM_MODEL_PATH}"
echo " 结果: ${RESULT_FILE}"
echo "============================================================"

python benchmark.py \
  --om_model_path "$OM_MODEL_PATH" \
  --hf_model_dir "$HF_MODEL_DIR" \
  --kv_cache_length $KV_CACHE_LENGTH \
  --max_prefill_length $MAX_PREFILL_LENGTH \
  --prompt "请详细介绍一下机器学习的基本概念和常用算法" \
  --max_new_tokens 100 \
  --rounds 10 \
  --warmup 0 \
  --label "${VERSION}" \
  --device_id $DEVICE_ID \
  2>&1 | tee "$RESULT_FILE"

echo ""
echo "[DONE] 结果已保存: $RESULT_FILE"
