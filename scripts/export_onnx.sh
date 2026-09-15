#!/bin/bash
# Step 1: PyTorch → ONNX 导出
# 用法: bash scripts/export_onnx.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 配置（按需修改）----
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="../models/${MODEL_NAME}"
KV_CACHE_LENGTH=4096
KV_CACHE_LAYOUT="BSHD"
DEVICE_STR="npu"
DTYPE="float16"
MODELING_VERSION="v10_gate_up_prefuse"

OUTPUT_DIR="opt_models/v5_fp16"
ONNX_MODEL_PATH="${OUTPUT_DIR}/onnx_raw/${MODEL_NAME}.onnx"
# ---- 配置结束 ----

mkdir -p "${OUTPUT_DIR}/onnx_raw"

echo "============================================================"
echo " [Step 1] PyTorch → ONNX 导出"
echo " Modeling: ${MODELING_VERSION}"
echo " Model:    ${HF_MODEL_DIR}"
echo " Output:   ${ONNX_MODEL_PATH}"
echo " KV Cache: ${KV_CACHE_LENGTH}, Layout: ${KV_CACHE_LAYOUT}"
echo " Device:   ${DEVICE_STR}, Dtype: ${DTYPE}"
echo "============================================================"

cp "export/modeling_qwen2_${MODELING_VERSION}.py" export/modeling_qwen2.py

python3 export/export_onnx.py \
  --hf_model_dir="$HF_MODEL_DIR" \
  --onnx_model_path="$ONNX_MODEL_PATH" \
  --kv_cache_length="$KV_CACHE_LENGTH" \
  --kv_cache_layout="$KV_CACHE_LAYOUT" \
  --device_str="$DEVICE_STR" \
  --dtype="$DTYPE"

echo "导出完成: ${ONNX_MODEL_PATH}"
