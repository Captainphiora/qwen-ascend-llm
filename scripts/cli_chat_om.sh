# export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64:$LD_LIBRARY_PATH
SESSION_TYPE="acl"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/home/chenxinji/models/${MODEL_NAME}"
MAX_OUTPUT_LENGTH=4096


MAX_PREFILL_LENGTH=1
OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om"
# OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_310b.om"
# OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_310b_v2.om"
# OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_sim.om"
# OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_65536_1.om"
# OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_32768_1.om"

MAX_INPUT_LENGTH=1024

SOC_VERSION=Ascend310B1
# DTYPE="float32"

TEMPERATURE=0.6
# TEMPERATURE=1

python3 ./cli_chat.py \
  --session_type=$SESSION_TYPE \
  --hf_model_dir=$HF_MODEL_DIR \
  --om_model_path=$OM_MODEL_PATH \
  --max_input_length=$MAX_INPUT_LENGTH \
  --max_output_length=$MAX_OUTPUT_LENGTH \
  --max_prefill_length=$MAX_PREFILL_LENGTH \
  --temperature=$TEMPERATURE
