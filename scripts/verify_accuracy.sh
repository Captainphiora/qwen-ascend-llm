#!/bin/bash
# 量化精度验证: 对比 FP16 ONNX 和 fake_quant ONNX
# 用法: bash scripts/verify_accuracy.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 配置（按需修改）----
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="../models/${MODEL_NAME}"

# FP16 baseline (v5 modeling 导出的 raw ONNX)
FP16_ONNX="opt_models/v5_fp16/onnx_raw/${MODEL_NAME}.onnx"
# fake_quant ONNX (AMCT 产物)
FAKEQUANT_ONNX="opt_models/v10_w8a8/amct/model_deploy_fake_quant_model.onnx"
LABEL="w8a8"
# ---- 配置结束 ----

echo "============================================================"
echo " 量化精度验证"
echo " FP16:       ${FP16_ONNX}"
echo " FakeQuant:  ${FAKEQUANT_ONNX}"
echo "============================================================"

python3 scripts/verify_fakequant_v2.py \
  --fp16_onnx="$FP16_ONNX" \
  --fakequant_onnx="${LABEL}=${FAKEQUANT_ONNX}" \
  --hf_model_dir="$HF_MODEL_DIR"
