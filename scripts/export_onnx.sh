#!/bin/bash

# DEVICE_STR=cpu
# DTYPE=float32
DEVICE_STR=npu
DTYPE=float16
# KV_CACHE_LENGTH=4096
KV_CACHE_LENGTH=65536
SIMPLIFY=true
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"

HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
ONNX_MODEL_PATH="./output/onnx_${MODEL_NAME}_${KV_CACHE_LENGTH}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"

echo "开始执行 ONNX 导出，时间: $(date '+%H:%M:%S')"
echo "DEVICE_STR:$DEVICE_STR"

python3 export/export_onnx.py \
  --device_str=$DEVICE_STR \
  --dtype=$DTYPE \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --kv_cache_length=$KV_CACHE_LENGTH \
  --simplify=$SIMPLIFY

echo "导出执行结束，时间: $(date '+%H:%M:%S')"


