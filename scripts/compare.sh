
# MODEL_NAME="Qwen2-0.5B-Instruct"
# MODEL_NAME="Qwen2-0.5B-Instruct"
# MODEL_NAME="Qwen2-0.5B-Instruct"
MODEL_NAME="Qwen2.5-1.5B-Instruct"
# MODEL_NAME="Qwen2.5-0.5B-Instruct"
# 

HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"

KV_CACHE_LENGTH=1024
MAX_PREFILL_LENGTH=1
# ONNX_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx2_Qwen2-0.5B-Instruct_1024/Qwen2-0.5B-Instruct_1024.onnx"
# OM_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/model_910_cann900/Qwen2-0.5B-Instruct_1024_1_910_9382.om"
# ONNX_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx2_Qwen2-0.5B-Instruct_1024/Qwen2-0.5B-Instruct_1024.onnx"
# OM_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/model_910_cann900/Qwen2-0.5B-Instruct_1024_1_910_9382.om"
# ONNX_MODEL_PATH="./output/onnx2_${MODEL_NAME}_${KV_CACHE_LENGTH}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
ONNX_MODEL_PATH="./output/onnx2_${MODEL_NAME}_${KV_CACHE_LENGTH}/${MODEL_NAME}_${KV_CACHE_LENGTH}_rectified.onnx"
# OM_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/model_910_cann900/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}_910_9382.om"
# OM_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/model_910_cann900/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}_exmatmul.om"
OM_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/model_910_cann900/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}_rectified.om"
# conda run -n qwen_ascend_cann900 python3 export/compare.py \
echo ONNX_MODEL_PATH:$ONNX_MODEL_PATH
echo OM_MODEL_PATH:$OM_MODEL_PATH
python3 export/compare.py \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --om_model_path=$OM_MODEL_PATH \
  --kv_cache_length=$KV_CACHE_LENGTH \
  --cpu_thread=1 \
  --dtype="float16" \
  --max_prefill_length=$MAX_PREFILL_LENGTH