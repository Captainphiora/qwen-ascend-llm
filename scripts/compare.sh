MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"

KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1

ONNX_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx2_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096_rectified.onnx"
OM_MODEL_PATH="output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_rectified.om"
# python3 export/compare_bak.py \
python3 export/compare.py \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --om_model_path=$OM_MODEL_PATH \
  --kv_cache_length=$KV_CACHE_LENGTH \
  --cpu_thread=1 \
  --dtype="float16" \
  --max_prefill_length=$MAX_PREFILL_LENGTH