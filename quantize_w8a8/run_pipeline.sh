#!/bin/bash
# =============================================================
# W8A8 Quantization Pipeline - End to End
# Step 1: msmodelslim PTQ quantize
# Step 2: export/export_onnx.py (simplify=false)
# Step 3: export/change_node.py
# Step 4: export/onnx2om.py (precision_mode=mixed_float16)
# =============================================================
set -e

source ~/.bashrc
source ~/.bashrc_cann900
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate qwen_ascend_cann900
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

# ============ Configuration ============
MODEL_PATH="/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
QUANT_DIR="${PROJECT_DIR}/output/quant_w8a8"
ONNX_DIR="${PROJECT_DIR}/output/onnx_w8a8"
ONNX2_DIR="${PROJECT_DIR}/output/onnx2_w8a8"
CALIB_FILE="${PROJECT_DIR}/quantize_w8a8/calib_data/boolq.jsonl"

KV_CACHE_LENGTH=4096
DEVICE_ID=0
SOC_VERSION="auto"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
ONNX_MODEL_NAME="${MODEL_NAME}_w8a8_${KV_CACHE_LENGTH}.onnx"

PYTHON="/root/miniconda3/envs/qwen_ascend_cann900/bin/python"

echo "============================================"
echo " W8A8 Quantization Pipeline"
echo " Model: $(basename ${MODEL_PATH})"
echo " KV Cache: ${KV_CACHE_LENGTH}"
echo "============================================"

# ============ Step 1: PTQ Quantize ============
echo ""
echo "[Step 1/4] PTQ Quantization (W8A8 + AntiOutlier m4)..."
if [ -d "${QUANT_DIR}" ] && ls "${QUANT_DIR}"/*.safetensors 1>/dev/null 2>&1; then
    echo "  -> Quantized model already exists, skipping. Delete ${QUANT_DIR} to re-run."
else
    ${PYTHON} ${PROJECT_DIR}/quantize_w8a8/step1_quantize.py \
        --model_path "${MODEL_PATH}" \
        --save_directory "${QUANT_DIR}" \
        --device_type npu \
        --device_id ${DEVICE_ID} \
        --calib_file "${CALIB_FILE}" \
        --num_calibration_samples 50 \
        --w_bit 8 \
        --a_bit 8 \
        --anti_method m4 \
        --act_method 1 \
        --disable_names lm_head
fi

# ============ Step 2: Export ONNX ============
echo ""
echo "[Step 2/4] Export quantized model to ONNX (npu, float16, quantize=W8A8)..."
ONNX_PATH="${ONNX_DIR}/${ONNX_MODEL_NAME}"
mkdir -p "${ONNX_DIR}"
${PYTHON} ${PROJECT_DIR}/export/export_onnx.py \
    --device_str npu \
    --dtype float16 \
    --hf_model_dir "${QUANT_DIR}" \
    --onnx_model_path "${ONNX_PATH}" \
    --kv_cache_length ${KV_CACHE_LENGTH} \
    --simplify false \
    --quantize W8A8

# ============ Step 3: Change Node ============
echo ""
echo "[Step 3/4] Running change_node.py..."
ONNX2_PATH="${ONNX2_DIR}/${ONNX_MODEL_NAME}"
${PYTHON} ${PROJECT_DIR}/export/change_node.py \
    --input_model_path "${ONNX_PATH}" \
    --output_model_path "${ONNX2_PATH}"

# ============ Step 4: Compile .om ============
echo ""
echo "[Step 4/4] Compile ONNX to .om via ATC (precision_mode=mixed_float16)..."
OM_OUTPUT="${PROJECT_DIR}/output/om_w8a8/${MODEL_NAME}_w8a8_${KV_CACHE_LENGTH}"
mkdir -p "$(dirname ${OM_OUTPUT})"
${PYTHON} ${PROJECT_DIR}/export/onnx2om.py \
    --soc_version ${SOC_VERSION} \
    --hf_model_dir "${MODEL_PATH}" \
    --onnx_model_path "${ONNX2_PATH}" \
    --om_model_path "${OM_OUTPUT}" \
    --kv_cache_length ${KV_CACHE_LENGTH} \
    --max_prefill_length 1 \
    --precision_mode mixed_float16

# ============ Done ============
echo ""
echo "============================================"
echo " Pipeline Complete!"
echo " Quantized model: ${QUANT_DIR}"
echo " ONNX (raw):      ${ONNX_PATH}"
echo " ONNX (changed):  ${ONNX2_PATH}"
echo " OM model:        ${OM_OUTPUT}.om"
echo ""
echo " To run inference:"
echo "   python cli_chat.py --session_type acl --om_model_path ${OM_OUTPUT}.om --hf_model_dir ${MODEL_PATH}"
echo "============================================"
