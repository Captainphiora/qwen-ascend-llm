# INPUT="output/onnx2_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096_rectified.onnx"
INPUT="output/onnx_test_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx"
OUTPUT="output/onnx2_DeepSeek-R1-Distill-Qwen-1.5B_4096_simplified/DeepSeek-R1-Distill-Qwen-1.5B_4096_rectified_sim.onnx"
python export/simplify_onnx.py \
    --input $INPUT \
    --output $OUTPUT