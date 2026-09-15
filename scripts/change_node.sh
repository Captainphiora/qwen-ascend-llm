#!/bin/bash
# Step 2: ONNX 图优化（RoPE 融合）
# 用法: bash scripts/change_node.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 配置（按需修改）----
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
# 统一使用 310B1 兼容版本（Trilu 修复，不做 RoPE 融合）
# 910 和 310B1 共用此脚本，RoPE 融合提升不大（<3%），不值得维护两套
CHANGE_NODE_SCRIPT="export/change_node_v4_noexpand_310b.py"

OUTPUT_DIR="opt_models/v5_fp16"
INPUT_MODEL_PATH="${OUTPUT_DIR}/onnx_raw/${MODEL_NAME}.onnx"
OUTPUT_MODEL_PATH="${OUTPUT_DIR}/onnx_changed/${MODEL_NAME}.onnx"
# ---- 配置结束 ----

mkdir -p "${OUTPUT_DIR}/onnx_changed"

echo "============================================================"
echo " [Step 2] ONNX 图优化"
echo " Script:  ${CHANGE_NODE_SCRIPT}"
echo " Input:   ${INPUT_MODEL_PATH}"
echo " Output:  ${OUTPUT_MODEL_PATH}"
echo "============================================================"

python3 "$CHANGE_NODE_SCRIPT" \
  --input_model_path="$INPUT_MODEL_PATH" \
  --output_model_path="$OUTPUT_MODEL_PATH"

echo "图优化完成: ${OUTPUT_MODEL_PATH}"
