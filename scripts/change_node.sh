
# INPUT_MODEL_PATH="./output/onnx/DeepSeek-R1-Distill-Qwen-1.5B.onnx"
# OUTPUT_MODEL_PATH="./output/onnx2/DeepSeek-R1-Distill-Qwen-1.5B_1024.onnx"

# MAX_OUTPUT_LENGTH=32
MAX_OUTPUT_LENGTH=1
# INPUT_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/onnx/Qwen2.5-0.5B-Instruct_${MAX_OUTPUT_LENGTH}.onnx"
# OUTPUT_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/onnx2/Qwen2.5-0.5B-Instruct_${MAX_OUTPUT_LENGTH}.onnx"
INPUT_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/onnx_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx"
OUTPUT_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/onnx2_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx"
# OUTPUT_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/onnx2/Qwen2.5-0.5B-Instruct_${MAX_OUTPUT_LENGTH}.onnx"

# INPUT_MODEL_PATH="./output/onnx/Qwen2.5-0.5B_chat_1024.onnx"
# OUTPUT_MODEL_PATH="./output/onnx2/Qwen2.5-0.5B_chat_1024.onnx"
# uv run export/change_node.py \
python3 export/change_node.py \
  --input_model_path=$INPUT_MODEL_PATH \
  --output_model_path=$OUTPUT_MODEL_PATH