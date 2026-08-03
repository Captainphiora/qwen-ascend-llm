#!/bin/bash
# ============================================================
# v1 RoPE 优化：ONNX → OM 编译
# 前置条件: 已执行 scripts/export_onnx_v1_rope.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."

VERSION="v1_rope"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
CPU_THREAD=64

ONNX_DIR="./output/onnx_${VERSION}"
ONNX_MODEL_PATH="${ONNX_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
OM_MODEL_PATH="./output/model_${VERSION}/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}"

# 如果有 simplify 版本，优先使用
ONNX_SIM="${ONNX_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}_sim.onnx"
if [ -f "$ONNX_SIM" ]; then
    ONNX_MODEL_PATH="$ONNX_SIM"
    echo "[INFO] 使用 simplified ONNX: $ONNX_SIM"
fi

# 如果需要 change_node
ONNX2_DIR="./output/onnx2_${VERSION}"
echo "[INFO] 执行 change_node..."
python3 export/change_node.py \
  --input_model_path="$ONNX_MODEL_PATH" \
  --output_model_path="${ONNX2_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}_rectified.onnx"
ONNX_MODEL_PATH="${ONNX2_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}_rectified.onnx"

echo ""
echo "============================================================"
echo " OM 编译: ${VERSION}"
echo " ONNX: ${ONNX_MODEL_PATH}"
echo " OM 输出: ${OM_MODEL_PATH}"
echo "============================================================"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python3 export/onnx2om.py \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --om_model_path=$OM_MODEL_PATH \
  --kv_cache_length=$KV_CACHE_LENGTH \
  --cpu_thread=$CPU_THREAD \
  --max_prefill_length=$MAX_PREFILL_LENGTH

echo ""
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[DONE] OM 已编译"
echo ""
echo " 下一步: bash scripts/bench_v1_rope.sh"
