#!/bin/bash
# 方案B-1: W8A16 权重量化 + OM编译
# 将Linear层权重从FP16量化为INT8, 激活保持FP16
# 预期: 模型减小~36%, 带宽压力降低, decode阶段可能加速

set -e

CONDA_ENV="qwen_ascend_cann900"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=8
CPU_THREAD=64

HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
INPUT_ONNX="opt_models/v4_noexpand/onnx_changed/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
OUTPUT_ONNX="output/onnx_quantized/${MODEL_NAME}_${KV_CACHE_LENGTH}_w8a16.onnx"
OM_MODEL_PATH="output/om_v4_noexpand_w8a16/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}"

echo "============================================"
echo " Step 1: W8A16 Weight Quantization"
echo "============================================"
conda run -n ${CONDA_ENV} env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python3 export/quantize/quantize_weights.py \
    --input_model "${INPUT_ONNX}" \
    --output_model "${OUTPUT_ONNX}"

echo ""
echo "============================================"
echo " Step 2: Compile W8A16 ONNX to OM"
echo "============================================"
conda run -n ${CONDA_ENV} \
  python3 export/onnx2om.py \
    --hf_model_dir=${HF_MODEL_DIR} \
    --onnx_model_path="${OUTPUT_ONNX}" \
    --om_model_path="${OM_MODEL_PATH}" \
    --kv_cache_length=${KV_CACHE_LENGTH} \
    --cpu_thread=${CPU_THREAD} \
    --max_prefill_length=${MAX_PREFILL_LENGTH}

echo ""
echo "============================================"
echo " Done!"
echo "============================================"
echo "  FP16 OM:  output/om_v4_noexpand/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}.om"
echo "  W8A16 OM: ${OM_MODEL_PATH}.om"
echo ""
echo "Next: run benchmark to compare performance"
