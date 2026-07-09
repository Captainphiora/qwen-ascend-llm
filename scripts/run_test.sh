#!/bin/bash

# DEVICE_STR=cpu
# DTYPE=float32
DEVICE_STR=npu
DTYPE=float16
KV_CACHE_LENGTH=1024
# KV_CACHE_LENGTH=2048
# KV_CACHE_LENGTH=16384
# KV_CACHE_LENGTH=32768
MAX_PREFILL_LENGTH=1

# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME="Qwen2.5-0.5B-Instruct"
MODEL_NAME="Qwen2-0.5B-Instruct"
# MODEL_NAME="Qwen2-1.5B-Instruct"
# MODEL_NAME="Qwen2.5-1.5B-Instruct"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
ONNX_MODEL_PATH="./output/onnx/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}.onnx"


echo "开始执行 ONNX 导出，时间: $(date '+%H:%M:%S')"
echo "DEVICE_STR:$DEVICE_STR"
# uv run export/export_onnx.py \
python3 export/export_onnx.py \
  --device_str=$DEVICE_STR \
  --dtype=$DTYPE \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --kv_cache_length=$KV_CACHE_LENGTH

# echo "导出执行结束，时间: $(date '+%H:%M:%S')"
# INTPUT_MODEL_PATH="./output/onnx/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
OUTPUT_MODEL_PATH="./output/onnx2/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"

# uv run export/change_node.py \
python3 export/change_node.py \
  --input_model_path=$ONNX_MODEL_PATH \
  --output_model_path=$OUTPUT_MODEL_PATH

# ONNX_MODEL_PATH="./output/onnx2_${KV_CACHE_LENGTH}/DeepSeek-R1-Distill-Qwen-1.5B_${KV_CACHE_LENGTH}.onnx"
ONNX_MODEL_PATH=${OUTPUT_MODEL_PATH}
# echo "ONNX_MODEL_PATH:$ONNX_MODEL_PATH"
OM_MODEL_PATH="./output/model_910_cann900/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}"
# OM_MODEL_PATH="./output/model_910/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}"
MAX_INPUT_LENGTH=1024
CPU_THREAD=64
# SOC_VERSION=Ascend310B1



# conda run -n llm_test python3 export/onnx2om.py \
# uv run export/onnx2om.py \
python3 export/onnx2om.py \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --om_model_path=$OM_MODEL_PATH \
  --kv_cache_length=$KV_CACHE_LENGTH \
  --cpu_thread=$CPU_THREAD \
  --max_prefill_length=$MAX_PREFILL_LENGTH
  # --soc_version=$SOC_VERSION
