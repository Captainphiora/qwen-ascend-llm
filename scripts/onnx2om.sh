TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG_DIR="./log/run_${TIMESTAMP}"

mkdir -p "${RUN_LOG_DIR}"
# 可选值说明: 0(DEBUG,最详细), 1(INFO), 2(WARNING), 3(ERROR)
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH="$(pwd)/${RUN_LOG_DIR}"

HF_MODEL_DIR="/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B"
# KV_CACHE_LENGTH=1024
KV_CACHE_LENGTH=4096
CPU_THREAD=1
MAX_PREFILL_LENGTH=1
SOC_VERSION=Ascend310B1
# ONNX_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/onnx2_ds_qwen/DeepSeek-R1-Distill-Qwen-1.5B_1024.onnx"
# OM_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/model/310b1_DeepSeek-R1-Distill-Qwen-1.5B_1024"
ONNX_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/onnx2_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096_rectified.onnx"
# OM_MODEL_PATH="/home/chenxinji/qwen-ascend-llm/output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_310b"
OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_310b_v3"

echo "=================================================="
echo "开始转换模型..."
echo "本次运行控制台日志: ${RUN_LOG_DIR}/onnx2om_console.log"
echo "本次运行底层 plog 目录: ${RUN_LOG_DIR}/"
echo "=================================================="

python3 export/onnx2om.py \
  --hf_model_dir=$HF_MODEL_DIR \
  --onnx_model_path=$ONNX_MODEL_PATH \
  --om_model_path=$OM_MODEL_PATH \
  --kv_cache_length=$KV_CACHE_LENGTH \
  --cpu_thread=$CPU_THREAD \
  --max_prefill_length=$MAX_PREFILL_LENGTH \
  --soc_version=$SOC_VERSION 2>&1 | tee "${RUN_LOG_DIR}/onnx2om_console.log"