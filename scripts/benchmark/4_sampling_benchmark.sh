#!/bin/bash
# ============================================================
# 脚本4: 不同采样方式推理测试
# 对比 greedy / top_p / top_k 以及 CPU / NPU 采样的性能差异
#
# 使用方式:
#   bash scripts/4_sampling_benchmark.sh                    # 默认 CPU 采样
#   bash scripts/4_sampling_benchmark.sh --npu-sampling     # 含 NPU 零拷贝采样
#   bash scripts/4_sampling_benchmark.sh --device_id=5
#   bash scripts/4_sampling_benchmark.sh --tokens=200 --rounds=5
#
# 示例:
#   # CPU 采样性能对比 (greedy/top_p/top_k)
#   bash scripts/4_sampling_benchmark.sh --device_id=7 --tokens=100 --rounds=3
#
#   # 含 NPU 零拷贝采样对比
#   bash scripts/4_sampling_benchmark.sh --device_id=7 --npu-sampling
#
#   # 快速测试
#   bash scripts/4_sampling_benchmark.sh --device_id=7 --tokens=30 --rounds=1
#
# 输出:
#   日志: scripts/logs/4_sampling_benchmark_<timestamp>.log
#
# 测试配置:
#   - Greedy (CPU argmax)
#   - Top-p=0.8 (CPU numpy)
#   - Top-p=0.95 (CPU numpy)
#   - Top-k=50 (CPU numpy)
#   - [可选] Top-p=0.8 (NPU ATB 零拷贝)
#   - [可选] Top-k=50 (NPU ATB 零拷贝)
# ============================================================

set -e
# source ~/.bashrc_cann900
source ~/.bashrc

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# 低内存模式: 使用 acl.mdl.load_from_file 避免分配与OM等大的pinned host内存
export ACL_LOAD_FROM_FILE=1

# ---- 默认配置 ----
DEVICE_ID=0
OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om"
HF_MODEL_DIR="/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
MAX_NEW_TOKENS=100
# ROUNDS=3
ROUNDS=3
# WARMUP=1
WARMUP=1
PROMPT="请详细介绍一下机器学习的基本概念和常用算法"
NPU_SAMPLING=false
# ---- 配置结束 ----

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --device_id=*) DEVICE_ID="${arg#*=}" ;;
        --tokens=*) MAX_NEW_TOKENS="${arg#*=}" ;;
        --rounds=*) ROUNDS="${arg#*=}" ;;
        --npu-sampling) NPU_SAMPLING=true ;;
        --om=*) OM_MODEL_PATH="${arg#*=}" ;;
        --help|-h)
            sed -n '2,35p' "$0"
            exit 0
            ;;
    esac
done

if [ "$NPU_SAMPLING" = true ]; then
    export USE_NPU_SAMPLING=1
else
    export USE_NPU_SAMPLING=0
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="scripts/logs/4_sampling_benchmark_${TIMESTAMP}.log"
mkdir -p scripts/logs

echo "============================================================" | tee "$LOG_FILE"
echo " [4] 采样方式性能对比" | tee -a "$LOG_FILE"
echo " Device: npu:${DEVICE_ID}" | tee -a "$LOG_FILE"
echo " Model: ${OM_MODEL_PATH}" | tee -a "$LOG_FILE"
echo " NPU Sampling: ${NPU_SAMPLING}" | tee -a "$LOG_FILE"
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
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 完成! 日志: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
