source ~/.bashrc_uv_cann82 

SESSION_TYPE="onnx"


DEVICE_STR=npu
# DEVICE_STR=cpu
# KV_CACHE_LENGTH=2048
KV_CACHE_LENGTH=1024
# MODEL_NAME="Qwen2.5-0.5B-Instruct"
# MODEL_NAME="Qwen2.5-1.5B-Instruct"
# MODEL_NAME="Qwen2.5-1.5B-Instruct"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
DTYPE="float16"
# DTYPE="float32"
MAX_INPUT_LENGTH=1024
# KV_CACHE_LENGTH=1024
CPU_THREAD=8
MAX_OUTPUT_LENGTH=${KV_CACHE_LENGTH}
ONNX_MODEL_PATH="./output/onnx/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"
# ONNX_MODEL_PATH="./output/onnx_qwen2.5_2048_npu/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"
# ONNX_MODEL_PATH="./output/onnx_ds_qwen_2048_npu/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"
# ONNX_MODEL_PATH="./output/onnx2_qwen2.5_2048_npu/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"
# ONNX_MODEL_PATH="./output/onnx_qwen2.5_2048_npu/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"
# ONNX_MODEL_PATH="./output/onnx_qwen2.5_2048_cpu/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"

uv run ./cli_chat.py \
  --session_type=$SESSION_TYPE \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --cpu_thread=$CPU_THREAD \
  --dtype=$DTYPE \
  --max_input_length=$MAX_INPUT_LENGTH \
  --max_output_length=$MAX_OUTPUT_LENGTH
