#!/bin/bash
# AMCT W8A8 校准: 对 FP16 ONNX 做 PTQ 量化，产出 deploy + fake_quant ONNX
# 用法: bash scripts/amct_calibrate.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 配置（按需修改）----
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="../models/${MODEL_NAME}"
KV_CACHE_LENGTH=4096
KV_CACHE_LAYOUT="BSHD"

# 输入: v10 modeling 导出的 FP16 ONNX
INPUT_ONNX="opt_models/v10_w8a8/onnx_raw/${MODEL_NAME}.onnx"
# 输出目录
OUTPUT_DIR="opt_models/v10_w8a8/amct"

QUANT_CFG="scripts/quant_v8_skip60_gate_up_quantized.cfg"
CALIB_FILE="/usr/local/Ascend/atb-models/examples/convert/model_slim/boolq.jsonl"
NUM_SAMPLES=50
CPU_THREADS=64
# ---- 配置结束 ----

echo "============================================================"
echo " AMCT W8A8 校准"
echo " Input:    ${INPUT_ONNX}"
echo " Config:   ${QUANT_CFG}"
echo " Samples:  ${NUM_SAMPLES}"
echo " Output:   ${OUTPUT_DIR}/"
echo "============================================================"

python3 scripts/amct_onnx_calibrate.py \
  --model_path="$INPUT_ONNX" \
  --hf_model_dir="$HF_MODEL_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --calib_file="$CALIB_FILE" \
  --num_samples="$NUM_SAMPLES" \
  --kv_cache_length="$KV_CACHE_LENGTH" \
  --cpu_threads="$CPU_THREADS" \
  --kv_cache_layout="$KV_CACHE_LAYOUT" \
  --quant_cfg="$QUANT_CFG"

echo ""
echo "校准完成:"
echo "  Deploy ONNX:     ${OUTPUT_DIR}/model_deploy_deploy_model.onnx"
echo "  FakeQuant ONNX:  ${OUTPUT_DIR}/model_deploy_fake_quant_model.onnx"
echo ""
echo "下一步:"
echo "  1. (可选) 精度验证: bash scripts/verify_accuracy.sh"
echo "  2. 图优化: bash scripts/change_node.sh  (需修改 INPUT 为 deploy ONNX)"
echo "  3. ATC 编译: bash scripts/onnx2om.sh    (需加 --precision_mode origin)"
