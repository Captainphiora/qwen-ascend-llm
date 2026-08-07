#!/bin/bash
# 方案B-2: atc Calibration PTQ (训练后量化)
# 使用atc内置的--compression_optimize_conf进行INT8校准量化
# 同时量化权重和激活, 理论上性能提升最大

set -e

CONDA_ENV="qwen_ascend_cann900"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=8
CPU_THREAD=64
NUM_CALIB_SAMPLES=8

HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
INPUT_ONNX="opt_models/v4_noexpand/onnx_changed/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
CALIB_DATA_DIR="output/calibration_data"
OM_MODEL_PATH="output/om_v4_noexpand_ptq/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}"
INFER_SOC="Ascend910_9382"

echo "============================================"
echo " Step 1: Generate calibration data"
echo "============================================"
conda run -n ${CONDA_ENV} env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python3 export/quantize/gen_calibration_data.py \
    --hf_model_dir "${HF_MODEL_DIR}" \
    --output_dir "${CALIB_DATA_DIR}" \
    --kv_cache_length ${KV_CACHE_LENGTH} \
    --num_samples ${NUM_CALIB_SAMPLES} \
    --infer_soc "${INFER_SOC}"

echo ""
echo "============================================"
echo " Step 2: Compile ONNX to OM with calibration PTQ"
echo "============================================"
conda run -n ${CONDA_ENV} \
  python3 export/onnx2om.py \
    --hf_model_dir=${HF_MODEL_DIR} \
    --onnx_model_path="${INPUT_ONNX}" \
    --om_model_path="${OM_MODEL_PATH}" \
    --kv_cache_length=${KV_CACHE_LENGTH} \
    --cpu_thread=${CPU_THREAD} \
    --max_prefill_length=${MAX_PREFILL_LENGTH} \
    --compression_optimize_conf="${CALIB_DATA_DIR}/compression_optimize.cfg"

echo ""
echo "============================================"
echo " Done!"
echo "============================================"
echo "  FP16 OM: output/om_v4_noexpand/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}.om"
echo "  PTQ OM:  ${OM_MODEL_PATH}.om"
echo ""
echo "Next: run benchmark to compare performance"
