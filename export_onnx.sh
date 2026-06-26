python3 export/export_onnx.py \
  --device_str=npu \
  --dtype=float16 \
  --hf_model_dir="/home/chenxinji/models/Qwen2.5-0.5B-Instruct"\
  --onnx_model_path="./output/onnx/qwen2.5_0.5b_chat.onnx" \
  --kv_cache_length=2048
