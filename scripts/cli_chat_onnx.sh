# source ~/.bashrc_uv_cann82 

SESSION_TYPE="onnx"
KV_CACHE_LENGTH=4096
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
# HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B-OrangePi-W8A8/deepseek-qwen-1.5B-w8a8"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
DEVICE_STR=npu
DTYPE="float16"
# DEVICE_STR=cpu
# DTYPE="float32"
MAX_INPUT_LENGTH=1024
CPU_THREAD=8
MAX_OUTPUT_LENGTH=${KV_CACHE_LENGTH}
TEMPERATURE=0
ONNX_MODEL_PATH="output/910_w8a8_noexpand/onnx_raw/DeepSeek-R1-Distill-Qwen-1.5B-OrangePi-W8A8/deepseek-qwen-1.5B-w8a8_4096.onnx"
# uv run ./cli_chat.py \
python3 ./cli_chat.py \
  --session_type=$SESSION_TYPE \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --cpu_thread=$CPU_THREAD \
  --device_str=$DEVICE_STR \
  --dtype=$DTYPE \
  --max_input_length=$MAX_INPUT_LENGTH \
  --max_output_length=$MAX_OUTPUT_LENGTH \
  --temperature=$TEMPERATURE
