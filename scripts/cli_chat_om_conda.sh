SESSION_TYPE="acl"
export ASCEND_DEVICE_ID=1

source ~/.bashrc_uv_cann82 

# HF_MODEL_DIR="/mnt/host-model/cxj/models/Qwen2.5-0.5B-Instruct"
# HF_MODEL_DIR="/mnt/host-model/cxj/models/Qwen2.5-1.5B-Instruct"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
MAX_INPUT_LENGTH=1024
MAX_OUTPUT_LENGTH=1024 #kv_cache_length=1024

KV_CACHE_LENGTH=1024
MAX_PREFILL_LENGTH=1
OM_MODEL_PATH="./output/model_910/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}.om"

CPU_THREAD=8
# SOC_VERSION=Ascend310B1
# DTYPE="float32"


# --cpu_thread=$CPU_THREAD \
  # --dtype=$DTYPE \
# uv run cli_chat.py \
conda run -n llm_test python3 cli_chat.py \
  --session_type=$SESSION_TYPE \
  --cpu_thread=$CPU_THREAD \
  --hf_model_dir=$HF_MODEL_DIR \
  --om_model_path=$OM_MODEL_PATH \
  --max_input_length=$MAX_INPUT_LENGTH \
  --max_output_length=$MAX_OUTPUT_LENGTH \
  --max_prefill_length=$MAX_PREFILL_LENGTH
