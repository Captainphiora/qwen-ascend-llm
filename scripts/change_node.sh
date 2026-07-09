
# INPUT_MODEL_PATH="./output/onnx/DeepSeek-R1-Distill-Qwen-1.5B.onnx"
# OUTPUT_MODEL_PATH="./output/onnx2/DeepSeek-R1-Distill-Qwen-1.5B_1024.onnx"


# MAX_OUTPUT_LENGTH=32 #kv_cache_length=1024
# INPUT_MODEL_PATH="./output/onnx/Qwen2.5-0.5B-Instruct_910_${MAX_OUTPUT_LENGTH}.onnx"
# OUTPUT_MODEL_PATH="./output/onnx2/Qwen2.5-0.5B-Instruct_910_${MAX_OUTPUT_LENGTH}.onnx"

# KV_CACHE_LENGTH=1024

# INPUT_MODEL_PATH="./output/onnx/Qwen2.5-0.5B_chat_${KV_CACHE_LENGTH}.onnx"
# OUTPUT_MODEL_PATH="./output/onnx2/Qwen2.5-0.5B_chat_${KV_CACHE_LENGTH}.onnx"

# MAX_OUTPUT_LENGTH=32
MAX_OUTPUT_LENGTH=1024
# MAX_OUTPUT_LENGTH=2048
# INPUT_MODEL_PATH="./output/onnx/DeepSeek-R1-Distill-Qwen-1.5B_${MAX_OUTPUT_LENGTH}.onnx"
# OUTPUT_MODEL_PATH="./output/onnx2/DeepSeek-R1-Distill-Qwen-1.5B_${MAX_OUTPUT_LENGTH}.onnx"
# INPUT_MODEL_PATH="./output/onnx_qwen2.5_npu/Qwen2.5-1.5B-Instruct_${MAX_OUTPUT_LENGTH}.onnx"
# OUTPUT_MODEL_PATH="./output/onnx2/Qwen2.5-1.5B-Instruct_${MAX_OUTPUT_LENGTH}.onnx"
# INPUT_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx_Qwen2.5-1.5B-Instruct_1024/Qwen2.5-1.5B-Instruct_1024.onnx"
# OUTPUT_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx2_Qwen2.5-1.5B-Instruct_1024/Qwen2.5-1.5B-Instruct_1024.onnx"
INPUT_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx_test_Qwen2.5-0.5B-Instruct_1024/Qwen2.5-0.5B-Instruct_1024.onnx"
# INPUT_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx_test_Qwen2.5-0.5B-Instruct_1024/Qwen2.5-0.5B-Instruct_1024.onnx"

OUTPUT_MODEL_PATH="/mnt/host-model/cxj/qwen-ascend-llm/output/onnx2_test_Qwen2.5-0.5B-Instruct_1024/Qwen2.5-0.5B-Instruct_1024.onnx"

# uv run export/change_node.py \
# python3 export/change_node.py \
conda run -n qwen_ascend_cann900 python3 export/change_node.py \
  --input_model_path=$INPUT_MODEL_PATH \
  --output_model_path=$OUTPUT_MODEL_PATH