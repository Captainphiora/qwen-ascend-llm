python3 ./api.py \
  --session_type="acl" \
  --hf_model_dir="../models/DeepSeek-R1-Distill-Qwen-1.5B" \
  --om_model_path="output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_8.om" \
  --max_input_length=1024 \
  --max_output_length=4096 \
  --max_prefill_length=8 \
  --temperature=0.6 \
  --top_p=0.95