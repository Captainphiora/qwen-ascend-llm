MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME="Qwen2.5-1.5B-Instruct"
# MODEL_NAME="Qwen2.5-0.5B-Instruct"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
KV_CACHE_LENGTH=1024

HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
ONNX_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx2_${MODEL_NAME}_${KV_CACHE_LENGTH}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
MAX_PREFILL_LENGTH=1
OM_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/model_910_cann900/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}_910_9382.om"
echo "OM_MODEL_PATH:$OM_MODEL_PATH"

CPU_THREAD=8
DTYPE="float16"
MAX_NEW_TOKENS=1024
PROMPT="你是谁"
# PROMPT="背诵《出师表》"


TEACHER_FORCING=0

python export/compare_greedy.py \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --om_model_path=$OM_MODEL_PATH \
  --kv_cache_length=$KV_CACHE_LENGTH \
  --max_prefill_length=$MAX_PREFILL_LENGTH \
  --cpu_thread=$CPU_THREAD \
  --dtype=$DTYPE \
  --max_new_tokens=$MAX_NEW_TOKENS \
  --prompt="$PROMPT" \
  --teacher_forcing=$TEACHER_FORCING