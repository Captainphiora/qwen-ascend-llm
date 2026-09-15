#!/bin/bash
# Step 4: OM 模型推理（CLI 交互）
# 用法: bash scripts/cli_chat_om.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 配置（按需修改）----
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="../models/${MODEL_NAME}"
OM_MODEL_PATH="opt_models/v5_fp16/om/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om"

MAX_INPUT_LENGTH=1024
MAX_OUTPUT_LENGTH=4096
MAX_PREFILL_LENGTH=1
DTYPE="float16"

SAMPLING_METHOD="greedy"
SAMPLING_VALUE=0.95
TEMPERATURE=0

DEVICE_STR="npu"
DEVICE_ID=0
CPU_THREAD=8
# ---- 配置结束 ----

echo "============================================================"
echo " [Step 4] OM 推理"
echo " Model:   ${HF_MODEL_DIR}"
echo " OM:      ${OM_MODEL_PATH}"
echo " Sampling: ${SAMPLING_METHOD}, Temperature: ${TEMPERATURE}"
echo " Device:  ${DEVICE_STR}:${DEVICE_ID}"
echo "============================================================"

python3 cli_chat.py \
  --session_type=acl \
  --hf_model_dir="$HF_MODEL_DIR" \
  --om_model_path="$OM_MODEL_PATH" \
  --max_input_length="$MAX_INPUT_LENGTH" \
  --max_output_length="$MAX_OUTPUT_LENGTH" \
  --max_prefill_length="$MAX_PREFILL_LENGTH" \
  --dtype="$DTYPE" \
  --sampling_method="$SAMPLING_METHOD" \
  --sampling_value="$SAMPLING_VALUE" \
  --temperature="$TEMPERATURE" \
  --device_str="$DEVICE_STR" \
  --device_id="$DEVICE_ID" \
  --cpu_thread="$CPU_THREAD"
