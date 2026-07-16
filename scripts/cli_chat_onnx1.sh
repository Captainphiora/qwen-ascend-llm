# source ~/.bashrc_uv_cann82 

SESSION_TYPE="onnx"



# KV_CACHE_LENGTH=1024
# KV_CACHE_LENGTH=2048
KV_CACHE_LENGTH=4096
# MODEL_NAME="Qwen2-0.5B-Instruct"
# MODEL_NAME="Qwen2-1.5B-Instruct"
# MODEL_NAME="Qwen2.5-0.5B-Instruct"
# MODEL_NAME="Qwen2.5-1.5B-Instruct"
# MODEL_NAME="Qwen2.5-1.5B-Instruct"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
DEVICE_STR=npu
DTYPE="float16"
# DEVICE_STR=cpu
# DTYPE="float32"
MAX_INPUT_LENGTH=1024
# KV_CACHE_LENGTH=1024
CPU_THREAD=8
MAX_OUTPUT_LENGTH=${KV_CACHE_LENGTH}
TEMPERATURE=0
# ONNX_MODEL_PATH="./output/onnx/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"
# ONNX_MODEL_PATH="./output/onnx_qwen2.5_2048_npu/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"
# ONNX_MODEL_PATH="./output/onnx_ds_qwen_2048_npu/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"
# ONNX_MODEL_PATH="./output/onnx2_qwen2.5_2048_npu/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"
# ONNX_MODEL_PATH="./output/onnx_qwen2.5_2048_npu/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"
# ONNX_MODEL_PATH="./output/onnx_qwen2.5_2048_cpu/${MODEL_NAME}_${MAX_OUTPUT_LENGTH}.onnx"
# ONNX_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx_ds_qwen_1024_cpu/DeepSeek-R1-Distill-Qwen-1.5B_1024.onnx"
# ONNX_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx2_qwen2.5_2048_cpu/Qwen2.5-1.5B-Instruct_2048.onnx"
# ONNX_MODEL_PATH="./output/onnx_Qwen2.5-1.5B-Instruct_1024/Qwen2.5-1.5B-Instruct_1024.onnx"
# ONNX_MODEL_PATH="./output/onnx_Qwen2-1.5B-Instruct_1024/Qwen2-1.5B-Instruct_1024_1.onnx"
# ONNX_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx_Qwen2-0.5B-Instruct_1024/Qwen2-0.5B-Instruct_1024_1.onnx"
# ONNX_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx_Qwen2.5-1.5B-Instruct_1024/Qwen2.5-1.5B-Instruct_1024_1.onnx"
ONNX_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx"
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
