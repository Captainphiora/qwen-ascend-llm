#!/bin/bash
# Step 3: ATC 编译（ONNX → OM）— W8A8, Ascend310B1
# 用法: bash scripts/onnx2om_w8a8_310b1.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 配置 ----
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="../models/${MODEL_NAME}"
KV_CACHE_LENGTH=4096
KV_CACHE_LAYOUT="BSHD"
MAX_PREFILL_LENGTH=1
SOC_VERSION="Ascend310B1"
CPU_THREAD=32
ONNX_MODEL_PATH="opt_models/v10_w8a8/onnx_changed/${MODEL_NAME}.onnx"
OM_MODEL_PATH="opt_models/v10_w8a8/om_310b1/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}"
# ---- 配置结束 ----

mkdir -p "opt_models/v10_w8a8/om_310b1"

echo "============================================================"
echo " [Step 3] ATC 编译 — W8A8 (Ascend310B1, precision_mode=origin)"
echo " Input:   ${ONNX_MODEL_PATH}"
echo " Output:  ${OM_MODEL_PATH}.om"
echo " SOC:     ${SOC_VERSION}"
echo "============================================================"

python3 export/onnx2om.py \
  --hf_model_dir="$HF_MODEL_DIR" \
  --onnx_model_path="$ONNX_MODEL_PATH" \
  --om_model_path="$OM_MODEL_PATH" \
  --kv_cache_length="$KV_CACHE_LENGTH" \
  --kv_cache_layout="$KV_CACHE_LAYOUT" \
  --max_prefill_length="$MAX_PREFILL_LENGTH" \
  --soc_version="$SOC_VERSION" \
  --precision_mode=origin \
  --cpu_thread=CPU_THREAD

echo "ATC 编译完成: ${OM_MODEL_PATH}.om"
