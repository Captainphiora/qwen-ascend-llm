SESSION_TYPE="acl"
# HF_MODEL_DIR="/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B"
# OM_MODEL_PATH="./output/model/DeepSeek-R1-Distill-Qwen-1.5B_1024_non_dynamic_310b1.om"
# HF_MODEL_DIR="/mnt/host-model/cxj/models/Qwen2.5-0.5B-Instruct"
HF_MODEL_DIR="/mnt/host-model/cxj/models/Qwen2.5-1.5B-Instruct"
MAX_INPUT_LENGTH=1024
MAX_OUTPUT_LENGTH=2048 #kv_cache_length=1024
# OM_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/model/Qwen2.5-0.5B-Instruct_${MAX_OUTPUT_LENGTH}_non_dynamic_310b1.om"
# OM_MODEL_PATH="./output/model/910_Qwen2.5-0.5B-Instruct_${MAX_OUTPUT_LENGTH}"
# OM_MODEL_PATH="./output/model/910_Qwen2.5-0.5B-Instruct_${MAX_OUTPUT_LENGTH}"
OM_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/model/Qwen2.5-1.5B-Instruct_2048_1_910.om"
CPU_THREAD=8
MAX_PREFILL_LENGTH=1
# SOC_VERSION=Ascend310B1
# DTYPE="float32"


# --cpu_thread=$CPU_THREAD \
  # --dtype=$DTYPE \
uv run ./cli_chat.py \
  --session_type=$SESSION_TYPE \
  --cpu_thread=$CPU_THREAD \
  --hf_model_dir=$HF_MODEL_DIR \
  --om_model_path=$OM_MODEL_PATH \
  --max_input_length=$MAX_INPUT_LENGTH \
  --max_output_length=$MAX_OUTPUT_LENGTH \
  --max_prefill_length=$MAX_PREFILL_LENGTH
