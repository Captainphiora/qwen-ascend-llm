#!/bin/bash
# ============================================================
# 脚本: 仅 NPU 采样性能测试
# 只运行 NPU ATB 零拷贝采样，跳过 CPU 采样
#
# 使用方式:
#   bash scripts/4_sampling_benchmark_npu_only.sh
#   bash scripts/4_sampling_benchmark_npu_only.sh --device_id=7
#   bash scripts/4_sampling_benchmark_npu_only.sh --device_id=7 --tokens=200 --rounds=5
#
# 输出:
#   日志: scripts/logs/4_sampling_benchmark_npu_only_<timestamp>.log
# ============================================================

set -e
source ~/.bashrc

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 默认配置 ----
DEVICE_ID=0
OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om"
HF_MODEL_DIR="../models/DeepSeek-R1-Distill-Qwen-1.5B"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
MAX_NEW_TOKENS=100
# ROUNDS=3
ROUNDS=1
WARMUP=0
# WARMUP=1
PROMPT="请详细介绍一下机器学习的基本概念和常用算法"
# ---- 配置结束 ----

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --device_id=*) DEVICE_ID="${arg#*=}" ;;
        --tokens=*) MAX_NEW_TOKENS="${arg#*=}" ;;
        --rounds=*) ROUNDS="${arg#*=}" ;;
        --om=*) OM_MODEL_PATH="${arg#*=}" ;;
        --help|-h)
            sed -n '2,14p' "$0"
            exit 0
            ;;
    esac
done

export USE_NPU_SAMPLING=1
export ACL_LOAD_FROM_FILE=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="scripts/logs/4_sampling_benchmark_npu_only_${TIMESTAMP}.log"
mkdir -p scripts/logs

echo "============================================================" | tee "$LOG_FILE"
echo " [4] NPU 采样性能测试 (仅NPU)" | tee -a "$LOG_FILE"
echo " Device: npu:${DEVICE_ID}" | tee -a "$LOG_FILE"
echo " Model: ${OM_MODEL_PATH}" | tee -a "$LOG_FILE"
echo " Tokens: ${MAX_NEW_TOKENS}, Rounds: ${ROUNDS}" | tee -a "$LOG_FILE"
echo " Time: ${TIMESTAMP}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python benchmark_sampling.py \
    --device_id "$DEVICE_ID" \
    --om_model_path "$OM_MODEL_PATH" \
    --hf_model_dir "$HF_MODEL_DIR" \
    --kv_cache_length "$KV_CACHE_LENGTH" \
    --max_prefill_length "$MAX_PREFILL_LENGTH" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --rounds "$ROUNDS" \
    --warmup "$WARMUP" \
    --npu-only \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 完成! 日志: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
