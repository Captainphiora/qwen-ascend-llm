#!/bin/bash
# ============================================================
# v2 RoPE + Expand 优化：导出 ONNX
# 前置条件: 已执行 python optimize_rope.py (包含 RoPE + Expand)
# ============================================================
set -e
cd "$(dirname "$0")/.."

VERSION="v2_rope_expand"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
KV_CACHE_LENGTH=4096
SIMPLIFY=false

ONNX_DIR="./output/onnx_${VERSION}"
ONNX_MODEL_PATH="${ONNX_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"

echo "============================================================"
echo " ONNX 导出: ${VERSION}"
echo " 输出: ${ONNX_DIR}"
echo "============================================================"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python3 export/export_onnx.py \
  --device_str=npu \
  --dtype=float16 \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --kv_cache_length=$KV_CACHE_LENGTH \
  --simplify=$SIMPLIFY

echo ""
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[DONE] ONNX 已导出: $ONNX_MODEL_PATH"
echo ""
echo " 下一步: bash scripts/onnx2om_v2_rope_expand.sh"
