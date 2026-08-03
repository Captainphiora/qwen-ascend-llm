#!/bin/bash
# ============================================================
# v2 RoPE + Expand 优化：性能测试
# 前置条件: 已执行 scripts/onnx2om_v2_rope_expand.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."

VERSION="v2_rope_expand"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
OM_MODEL_PATH="./output/model_${VERSION}/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}.om"

# 找到编译好的 OM
OM_DIR="./output/model_${VERSION}"
# OM_MODEL_PATH=$(find "$OM_DIR" -name "*.om" | head -1)
# if [ -z "$OM_MODEL_PATH" ]; then
#     echo "[ERROR] 在 $OM_DIR 下找不到 .om 文件"
#     echo "请先运行: bash scripts/onnx2om_v2_rope_expand.sh"
#     exit 1
# fi

RESULT_DIR="./benchmark_results"
mkdir -p "$RESULT_DIR"
RESULT_FILE="${RESULT_DIR}/${VERSION}.txt"

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
  --max_new_tokens 30 \
  --rounds 3 \
  --warmup 1 \
  --label "${VERSION}" \
  2>&1 | tee "$RESULT_FILE"

echo ""
echo "[DONE] 结果已保存: $RESULT_FILE"
