#!/bin/bash

KV_CACHE_LENGTH=4096
# KV_CACHE_LENGTH=32768
# KV_CACHE_LENGTH=65536
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"

HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
OUTPUT_MODEL_PATH="opt_models/v4_noexpand/onnx_changed/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx"
MAX_PREFILL_LENGTH=8
ONNX_MODEL_PATH=${OUTPUT_MODEL_PATH}
OM_MODEL_PATH="output/om_v4_noexpand/DeepSeek-R1-Distill-Qwen-1.5B_4096_8"

CPU_THREAD=64
# SOC_VERSION=Ascend310B1
python3 export/onnx2om.py \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --om_model_path=$OM_MODEL_PATH \
  --kv_cache_length=$KV_CACHE_LENGTH \
  --cpu_thread=$CPU_THREAD \
  --max_prefill_length=$MAX_PREFILL_LENGTH
  # --soc_version=$SOC_VERSION
