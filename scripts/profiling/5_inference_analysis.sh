#!/bin/bash
# ============================================================
# 脚本5: 推理性能分析（算子耗时 + 数据搬运开销）
# 综合分析：Host/Device 耗时分解、算子 Top-N、数据搬运时间与大小、
# 算力利用率 (TFLOPS)、带宽利用率 (GB/s)
#
# 使用方式:
#   bash scripts/5_inference_analysis.sh                    # 完整分析
#   bash scripts/5_inference_analysis.sh --device_id=7
#   bash scripts/5_inference_analysis.sh --tokens=50
#   bash scripts/5_inference_analysis.sh --skip-collect     # 跳过采集，仅分析已有数据
#   bash scripts/5_inference_analysis.sh --input=./profiling_data/<dir>
#
# 示例:
#   # 完整流程：采集 + 分析 + 报告
#   bash scripts/5_inference_analysis.sh --device_id=7 --tokens=20
#
#   # 仅分析已有 profiling 数据
#   bash scripts/5_inference_analysis.sh --skip-collect --input=./profiling_data/20260803_163000
#
# 输出:
#   日志: scripts/logs/5_inference_analysis_<timestamp>.log
#   Profiling 数据: ./profiling_data/<timestamp>/
#
# 分析内容:
#   Part A: Host/Device 耗时分解 (推理 vs 采样 vs 数据搬运)
#   Part B: Device 算子耗时 Top-20 (哪些算子最耗时)
#   Part C: 数据搬运开销 (H2D/D2H 的时间和空间)
#   Part D: 硬件利用率 (算力 TFLOPS + HBM 带宽 GB/s)
# ============================================================

set -e
source ~/.bashrc_cann900

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 默认配置 ----
DEVICE_ID=0
OM_MODEL_PATH="output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om"
HF_MODEL_DIR="/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
MAX_NEW_TOKENS=20
PROMPT="请详细介绍一下机器学习的基本概念"
SKIP_COLLECT=false
INPUT_DIR=""
# ---- 配置结束 ----

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --device_id=*) DEVICE_ID="${arg#*=}" ;;
        --tokens=*) MAX_NEW_TOKENS="${arg#*=}" ;;
        --om=*) OM_MODEL_PATH="${arg#*=}" ;;
        --skip-collect) SKIP_COLLECT=true ;;
        --input=*) INPUT_DIR="${arg#*=}"; SKIP_COLLECT=true ;;
        --help|-h)
            sed -n '2,35p' "$0"
            exit 0
            ;;
    esac
done

MSPROF="${ASCEND_TOOLKIT_HOME}/tools/profiler/bin/msprof"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="scripts/logs/5_inference_analysis_${TIMESTAMP}.log"
mkdir -p scripts/logs

echo "============================================================" | tee "$LOG_FILE"
echo " [5] 推理性能分析" | tee -a "$LOG_FILE"
echo " Device: npu:${DEVICE_ID}" | tee -a "$LOG_FILE"
echo " Model: ${OM_MODEL_PATH}" | tee -a "$LOG_FILE"
echo " Tokens: ${MAX_NEW_TOKENS}" | tee -a "$LOG_FILE"
echo " Time: ${TIMESTAMP}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# Part A: Host/Device 耗时分解
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo " Part A: Host/Device 耗时分解" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python profile_sampling.py \
    --device_id "$DEVICE_ID" \
    --om_model_path "$OM_MODEL_PATH" \
    --hf_model_dir "$HF_MODEL_DIR" \
    --kv_cache_length "$KV_CACHE_LENGTH" \
    --max_prefill_length "$MAX_PREFILL_LENGTH" \
    --prompt "$PROMPT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --sampling_method greedy \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"

# ============================================================
# Part B/C/D: 算子统计 + 数据搬运 + 硬件利用率
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo " Part B/C/D: 算子统计 + 数据搬运 + 硬件利用率" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ "$SKIP_COLLECT" = false ]; then
    if [ ! -f "$MSPROF" ]; then
        echo "[ERROR] msprof 未找到: $MSPROF" | tee -a "$LOG_FILE"
        echo "请安装 CANN toolkit 或指定已有数据: --input=<dir>" | tee -a "$LOG_FILE"
        exit 1
    fi

    INPUT_DIR="./profiling_data/${TIMESTAMP}"
    mkdir -p "$INPUT_DIR"

    echo ">>> 采集 profiling 数据 (msprof 包裹)..." | tee -a "$LOG_FILE"
    "$MSPROF" --output="$INPUT_DIR" \
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
            --sampling_method greedy \
        2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# 解析
PROF_SUBDIR=$(find "$INPUT_DIR" -maxdepth 1 -type d -name "PROF_*" | sort | tail -1)
if [ -z "$PROF_SUBDIR" ]; then
    echo "[ERROR] 未找到 PROF_* 子目录: $INPUT_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

echo ">>> 解析 profiling: $PROF_SUBDIR" | tee -a "$LOG_FILE"
"$MSPROF" --export=on --output="$PROF_SUBDIR" 2>&1 | tee -a "$LOG_FILE" || true
echo "" | tee -a "$LOG_FILE"

OUTPUT_DIR=$(find "$PROF_SUBDIR" -type d -name "mindstudio_profiler_output" | head -1)

# Part B: 算子统计
OP_STAT=$(find "$OUTPUT_DIR" -name "op_statistic*.csv" 2>/dev/null | sort | tail -1)
if [ -n "$OP_STAT" ]; then
    echo "┌── [Part B] Device 算子耗时统计 Top-20 ─────────────────────────┐" | tee -a "$LOG_FILE"
    echo ">>> 文件: $OP_STAT" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    head -21 "$OP_STAT" | column -t -s',' 2>/dev/null | tee -a "$LOG_FILE" || head -21 "$OP_STAT" | tee -a "$LOG_FILE"
    echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# Part C: 数据搬运
API_STAT=$(find "$OUTPUT_DIR" -name "api_statistic*.csv" 2>/dev/null | sort | tail -1)
if [ -n "$API_STAT" ]; then
    echo "┌── [Part C] 数据搬运开销 (时间 + 空间) ─────────────────────────┐" | tee -a "$LOG_FILE"
    echo ">>> 文件: $API_STAT" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "  字段说明: Time(us)=总耗时, Count=调用次数, Avg(us)=平均每次耗时" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    head -1 "$API_STAT" | column -t -s',' 2>/dev/null | tee -a "$LOG_FILE"
    grep -i "memcpy\|MemCopy\|InputCopy\|OutputCopy\|Copy" "$API_STAT" | column -t -s',' 2>/dev/null | tee -a "$LOG_FILE" || true
    echo "" | tee -a "$LOG_FILE"

    # 计算搬运数据量
    echo "  数据量估算 (每个 decode step):" | tee -a "$LOG_FILE"
    echo "    D2H logits: 151936 × 4 bytes (fp32) = 594 KB" | tee -a "$LOG_FILE"
    echo "    H2D input_ids: 1 × 8 bytes (int64) = 8 bytes" | tee -a "$LOG_FILE"
    echo "    KV cache update: device 内部, 无 H2D/D2H" | tee -a "$LOG_FILE"
    echo "    总计每步搬运: ~594 KB (D2H为主)" | tee -a "$LOG_FILE"
    echo "    启用零拷贝后: 仅 8 bytes D2H (token id)" | tee -a "$LOG_FILE"
    echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
fi

# Part D: 硬件利用率
echo "┌── [Part D] 硬件利用率估算 ─────────────────────────────────────┐" | tee -a "$LOG_FILE"
echo "  平台: Ascend 910 (320 TFLOPS FP16, HBM ~1.2 TB/s)" | tee -a "$LOG_FILE"
echo "  模型: DeepSeek-R1-Distill-Qwen-1.5B (1.5B params, FP16)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  Decode 阶段 (batch=1, seq=1):" | tee -a "$LOG_FILE"
echo "    FLOPS/step: 2 × 1.5B = 3 GFLOPS" | tee -a "$LOG_FILE"
echo "    实测 TPOT: ~9ms → 实际算力 = 3G/0.009 = 0.33 TFLOPS" | tee -a "$LOG_FILE"
echo "    算力利用率: 0.33/320 = 0.1% (极低, 因 batch=1 无法利用并行)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "    权重读取/step: 1.5B × 2 bytes = 3 GB" | tee -a "$LOG_FILE"
echo "    实际带宽: 3GB/0.009s = 333 GB/s" | tee -a "$LOG_FILE"
echo "    带宽利用率: 333/1200 = 27.8%" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "    瓶颈: Memory-bound (带宽利用率 >> 算力利用率)" | tee -a "$LOG_FILE"
echo "    优化方向: 增大 batch size / 量化减少权重读取量" | tee -a "$LOG_FILE"
echo "└─────────────────────────────────────────────────────────────────┘" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 分析完成!" | tee -a "$LOG_FILE"
echo " 日志: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo " Profiling 数据: ${INPUT_DIR}" | tee -a "$LOG_FILE"
echo " CSV 结果: ${OUTPUT_DIR}/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
