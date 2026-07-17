SESSION_TYPE="acl"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME="Qwen2-0.5B-Instruct"
# MODEL_NAME="Qwen2-1.5B-Instruct"
# MODEL_NAME="Qwen2.5-0.5B-Instruct"
# MODEL_NAME="Qwen2.5-1.5B-Instruct"
MAX_INPUT_LENGTH=1024
# MAX_OUTPUT_LENGTH=1024 #kv_cache_length=1024
MAX_OUTPUT_LENGTH=4096 #kv_cache_length=1024

# KV_CACHE_LENGTH=1024
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1

HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
# OM_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/model_910_cann900/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}
# _910_9382.om"
# OM_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/model_910_cann900/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}_exmatmul.om"
# OM_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/model_910_cann900/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}_rectified.om"
OM_MODEL_PATH="output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_sim.om"

echo "OM_MODEL_PATH:$OM_MODEL_PATH"

TEMPERATURE=0.6
# system_prompt
CPU_THREAD=8
# SOC_VERSION=Ascend310B1
# DTYPE="float32"

python3 cli_chat.py \
  --session_type=$SESSION_TYPE \
  --cpu_thread=$CPU_THREAD \
  --hf_model_dir=$HF_MODEL_DIR \
  --om_model_path=$OM_MODEL_PATH \
  --max_input_length=$MAX_INPUT_LENGTH \
  --max_output_length=$MAX_OUTPUT_LENGTH \
  --max_prefill_length=$MAX_PREFILL_LENGTH \
  --temperature=$TEMPERATURE
