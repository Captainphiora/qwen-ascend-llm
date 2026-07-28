SESSION_TYPE="onnx"
HF_MODEL_DIR="/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B"
# HF_MODEL_DIR="/home/chenxinji/models/Qwen2.5-0.5B-Instruct"
# MAX_OUTPUT_LENGTH=32 #kv_cache_length=1024
# MAX_OUTPUT_LENGTH=4096 #kv_cache_length=1024
MAX_INPUT_LENGTH=1024
# ONNX_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/onnx/Qwen2.5-0.5B-Instruct_${MAX_OUTPUT_LENGTH}.onnx"

ONNX_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/onnx_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx"
MAX_OUTPUT_LENGTH=4096

CPU_THREAD=1
MAX_PREFILL_LENGTH=1
DTYPE="float16"
DEVICE_STR="npu"
TEMPERATURE=0.6
SOC_VERSION=Ascend310B1
python ./cli_chat.py \
  --session_type=$SESSION_TYPE \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --dtype=$DTYPE \
  --cpu_thread=$CPU_THREAD \
  --max_input_length=$MAX_INPUT_LENGTH \
  --max_output_length=$MAX_OUTPUT_LENGTH \
  --max_prefill_length=$MAX_PREFILL_LENGTH \
  --device_str=$DEVICE_STR \
  --temperature=$TEMPERATURE
