SESSION_TYPE="acl"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"

MAX_INPUT_LENGTH=1024
MAX_OUTPUT_LENGTH=4096 #kv_cache_length=1024
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
HF_MODEL_DIR="../models/${MODEL_NAME}"
# OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om"
OM_MODEL_PATH="opt_models/v5_gate_up_fuse_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v5_gate_up_fuse_310b.om"
# OM_MODEL_PATH="opt_models/v6_transpose_elim_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v6_transpose_elim_310b.om"
# OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_w8a8_4096_1_w8a8_norope.om"
echo "OM_MODEL_PATH:$OM_MODEL_PATH"

SAMPLING_METHOD="greedy"
# SAMPLING_METHOD="top_p"
SAMPLING_VALUE=0.95
# SAMPLING_METHOD="top_k"
# SAMPLING_VALUE=40
TEMPERATURE=0.6
# TEMPERATURE=0
# system_prompt
CPU_THREAD=1
DEVICE_STR="npu"
DEVICE_ID=0
# SOC_VERSION=Ascend310B1
DTYPE="float16"
SAMPLING_DEVICE="cpu"
# SAMPLING_DEVICE="npu"
python3 cli_chat.py \
  --session_type=$SESSION_TYPE \
  --cpu_thread=$CPU_THREAD \
  --hf_model_dir=$HF_MODEL_DIR \
  --om_model_path=$OM_MODEL_PATH \
  --max_input_length=$MAX_INPUT_LENGTH \
  --max_output_length=$MAX_OUTPUT_LENGTH \
  --max_prefill_length=$MAX_PREFILL_LENGTH \
  --temperature=$TEMPERATURE \
  --dtype=$DTYPE \
  --sampling_method=$SAMPLING_METHOD \
  --sampling_value=$SAMPLING_VALUE \
  --device_str=$DEVICE_STR \
  --device_id=$DEVICE_ID \
  --sampling_device=$SAMPLING_DEVICE
