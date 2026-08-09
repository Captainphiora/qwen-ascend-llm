SESSION_TYPE="acl"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B-OrangePi-W8A8/deepseek-qwen-1.5B-w8a8"
MAX_INPUT_LENGTH=1024
MAX_OUTPUT_LENGTH=4096 #kv_cache_length=1024
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
HF_MODEL_DIR="../models/${MODEL_NAME}"
OM_MODEL_PATH="output/910_w8a8_noexpand/DeepSeek-R1-Distill-Qwen-1.5B-OrangePi-W8A8/deepseek-qwen-1.5B-w8a8_4096_1_910_w8a8_noexpand.om"

echo "OM_MODEL_PATH:$OM_MODEL_PATH"

SAMPLING_METHOD="greedy"
SAMPLING_VALUE=0.95
TEMPERATURE=0
CPU_THREAD=8
DEVICE_STR="npu"
DEVICE_ID=11
DTYPE="float16"

python3 cli_chat.py \
  --session_type=$SESSION_TYPE \
  --cpu_thread=$CPU_THREAD \
  --hf_model_dir=$HF_MODEL_DIR \
  --om_model_path=$OM_MODEL_PATH \
  --max_input_length=$MAX_INPUT_LENGTH \
  --max_output_length=$MAX_OUTPUT_LENGTH \
  --max_prefill_length=$MAX_PREFILL_LENGTH \
  --temperature=$TEMPERATURE \
  --dtype=$DTYPE \
  --sampling_method=$SAMPLING_METHOD \
  --sampling_value=$SAMPLING_VALUE \
  --device_str=$DEVICE_STR \
  --device_id=$DEVICE_ID
