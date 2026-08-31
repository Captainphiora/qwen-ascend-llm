#!/bin/bash
# ============================================================
# 脚本2: Profiling 数据采集
# 使用 msprof 包裹推理脚本，采集完整的 Host + Device profiling 数据
#
# 使用方式:
#   bash scripts/2_collect_profiling.sh                    # 默认配置
#   bash scripts/2_collect_profiling.sh --device_id=7      # 指定设备
#   bash scripts/2_collect_profiling.sh --tokens=50        # 指定生成 token 数
#   bash scripts/2_collect_profiling.sh --sampling=greedy  # 指定采样方式
#   bash scripts/2_collect_profiling.sh --sampling=top_p
#
# 示例:
#   # 采集 greedy 解码 20 tokens 的 profiling 数据
#   bash scripts/2_collect_profiling.sh --device_id=7 --tokens=20 --sampling=greedy
#
#   # 采集 top_p 解码的 profiling 数据
#   bash scripts/2_collect_profiling.sh --device_id=7 --tokens=20 --sampling=top_p
#
# 输出:
#   Profiling 原始数据: ./profiling_data/<timestamp>/
#   日志: scripts/logs/2_collect_profiling_<timestamp>.log
#
# 采集内容:
#   --ascendcl=on       Host 侧 ACL API 调用追踪
#   --task-time=on      Device 算子执行时间
#   --ai-core=on        AI Core 性能指标
#   --aic-metrics       流水线利用率
#   --aicpu=on          AI CPU 任务追踪
#   --runtime-api=on    Host runtime API 追踪
# ============================================================

set -e
source ~/.bashrc_cann900

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 默认配置 ----
DEVICE_ID=7
OM_MODEL_PATH="output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om"
HF_MODEL_DIR="/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
MAX_NEW_TOKENS=20
SAMPLING_METHOD="greedy"
PROMPT="请详细介绍一下机器学习的基本概念"
# ---- 配置结束 ----

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --device_id=*) DEVICE_ID="${arg#*=}" ;;
        --tokens=*) MAX_NEW_TOKENS="${arg#*=}" ;;
        --sampling=*) SAMPLING_METHOD="${arg#*=}" ;;
        --om=*) OM_MODEL_PATH="${arg#*=}" ;;
        --help|-h)
            sed -n '2,32p' "$0"
            exit 0
            ;;
    esac
done

MSPROF="${ASCEND_TOOLKIT_HOME}/tools/profiler/bin/msprof"
if [ ! -f "$MSPROF" ]; then
    echo "[ERROR] msprof 未找到: $MSPROF"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILING_DIR="./profiling_data/${TIMESTAMP}"
LOG_FILE="scripts/logs/2_collect_profiling_${TIMESTAMP}.log"
mkdir -p scripts/logs "$PROFILING_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo " [2] Profiling 数据采集" | tee -a "$LOG_FILE"
echo " Device: npu:${DEVICE_ID}" | tee -a "$LOG_FILE"
echo " Model: ${OM_MODEL_PATH}" | tee -a "$LOG_FILE"
echo " Sampling: ${SAMPLING_METHOD}" | tee -a "$LOG_FILE"
echo " Tokens: ${MAX_NEW_TOKENS}" | tee -a "$LOG_FILE"
echo " Output: ${PROFILING_DIR}" | tee -a "$LOG_FILE"
echo " msprof: ${MSPROF}" | tee -a "$LOG_FILE"
echo " Time: ${TIMESTAMP}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo ">>> 开始采集 profiling 数据..." | tee -a "$LOG_FILE"
"$MSPROF" --output="$PROFILING_DIR" \
    --ascendcl=on \
    --task-time=on \
    --ai-core=on \
    --aic-metrics=PipeUtilization \
    --aicpu=on \
    --runtime-api=on \
    python profile_sampling.py \
        --device_id "$DEVICE_ID" \
        --om_model_path "$OM_MODEL_PATH" \
        --hf_model_dir "$HF_MODEL_DIR" \
        --kv_cache_length "$KV_CACHE_LENGTH" \
        --max_prefill_length "$MAX_PREFILL_LENGTH" \
        --prompt "$PROMPT" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --sampling_method "$SAMPLING_METHOD" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 采集完成!" | tee -a "$LOG_FILE"
echo " Profiling 数据: ${PROFILING_DIR}" | tee -a "$LOG_FILE"
echo " 日志: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo " 下一步: 分析数据" | tee -a "$LOG_FILE"
echo "   bash scripts/3_analyze_profiling.sh --input=${PROFILING_DIR}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
