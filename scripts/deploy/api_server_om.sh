#!/bin/bash
# Launch OpenAI-compatible API server on Ascend 310B with .om model
# Usage: bash scripts/api_server_om.sh

SESSION_TYPE="acl"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/home/chenxinji/models/${MODEL_NAME}"
OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om"

MAX_INPUT_LENGTH=1024
MAX_OUTPUT_LENGTH=4096
MAX_PREFILL_LENGTH=1
TEMPERATURE=0.6

HOST="0.0.0.0"
PORT=1040

echo "============================================"
echo " Starting OpenAI-compatible API Server"
echo " Model : ${MODEL_NAME}"
echo " OM Path: ${OM_MODEL_PATH}"
echo " Listen : ${HOST}:${PORT}"
echo "============================================"

python3 api.py \
  --session_type=$SESSION_TYPE \
  --hf_model_dir=$HF_MODEL_DIR \
  --om_model_path=$OM_MODEL_PATH \
  --max_input_length=$MAX_INPUT_LENGTH \
  --max_output_length=$MAX_OUTPUT_LENGTH \
  --max_prefill_length=$MAX_PREFILL_LENGTH \
  --temperature=$TEMPERATURE
