#!/bin/bash

# set -e
source ~/.bashrc_uv_cann82 

KV_CACHE_LENGTH=512
# KV_CACHE_LENGTH=1024
# KV_CACHE_LENGTH=2048
# KV_CACHE_LENGTH=2048
# MODEL_NAME="Qwen2.5-1.5B-Instruct"
MODEL_NAME="Qwen2.5-0.5B-Instruct"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
OUTPUT_MODEL_PATH="./output/onnx2_${MODEL_NAME}_${KV_CACHE_LENGTH}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
MAX_PREFILL_LENGTH=1

ONNX_MODEL_PATH=${OUTPUT_MODEL_PATH}
# echo "ONNX_MODEL_PATH:$ONNX_MODEL_PATH"
OM_MODEL_PATH="./output/model/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}"
# OM_MODEL_PATH="./output/model_910_uv/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}"
MAX_INPUT_LENGTH=1024
CPU_THREAD=64
# SOC_VERSION=Ascend310B1
SOC_VERSION=Ascend310B1

# conda run -n llm_test python3 export/onnx2om.py \


# ds_distill_qwen_1.5b
# conda run -n ds_distill_qwen_1.5b python3 export/onnx2om.py \
# conda run -n llm_test python3 export/onnx2om.py \
uv run export/onnx2om.py \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --om_model_path=$OM_MODEL_PATH \
  --kv_cache_length=$KV_CACHE_LENGTH \
  --cpu_thread=$CPU_THREAD \
  --max_prefill_length=$MAX_PREFILL_LENGTH \
  --soc_version=$SOC_VERSION
