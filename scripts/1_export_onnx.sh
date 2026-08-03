#!/bin/bash
# ============================================================
# 脚本1: ONNX 模型导出
# 使用不同的 modeling_qwen2 文件替换原文件，导出 ONNX 模型
#
# 使用方式:
#   bash scripts/1_export_onnx.sh                          # 使用默认 (v4_noexpand)
#   bash scripts/1_export_onnx.sh --modeling=v4_noexpand   # 指定 modeling 版本
#   bash scripts/1_export_onnx.sh --modeling=baseline      # 使用原始 baseline
#   bash scripts/1_export_onnx.sh --modeling=v3_kvcache_noslice
#
# 可用 modeling 版本:
#   baseline              - 原始 modeling (export/modeling_qwen2.py)
#   v2_kvcache            - KV Cache 重构 (export/modeling_qwen2_v2_kvcache.py)
#   v3_kvcache_noslice    - KV Cache 6D 无 Slice (export/modeling_qwen2_v3_kvcache_noslice.py)
#   v4_noexpand           - KV Cache 6D + GQA broadcast (export/modeling_qwen2_v4_noexpand.py)
#
# 示例:
#   # 导出 v4_noexpand 优化版模型
#   bash scripts/1_export_onnx.sh --modeling=v4_noexpand --kv_cache_length=4096
#
#   # 导出 baseline 用于对比
#   bash scripts/1_export_onnx.sh --modeling=baseline --kv_cache_length=4096
#
# 输出:
#   ONNX 模型: ./output/onnx_<MODEL>_<KV_LEN>/<MODEL>_<KV_LEN>.onnx
#   日志: scripts/logs/1_export_onnx_<timestamp>.log
# ============================================================

set -e
source ~/.bashrc_cann900

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 默认配置 ----
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
KV_CACHE_LENGTH=4096
DEVICE_STR="npu"
DTYPE="float16"
SIMPLIFY="false"
MODELING_VERSION="v4_noexpand"
# ---- 配置结束 ----

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --modeling=*) MODELING_VERSION="${arg#*=}" ;;
        --kv_cache_length=*) KV_CACHE_LENGTH="${arg#*=}" ;;
        --simplify) SIMPLIFY="true" ;;
        --help|-h)
            sed -n '2,30p' "$0"
            exit 0
            ;;
    esac
done

# 映射 modeling 版本到文件
case "$MODELING_VERSION" in
    baseline)           MODELING_FILE="export/modeling_qwen2.py" ;;
    v2_kvcache)         MODELING_FILE="export/modeling_qwen2_v2_kvcache.py" ;;
    v3_kvcache_noslice) MODELING_FILE="export/modeling_qwen2_v3_kvcache_noslice.py" ;;
    v4_noexpand)        MODELING_FILE="export/modeling_qwen2_v4_noexpand.py" ;;
    *)
        echo "[ERROR] 未知 modeling 版本: $MODELING_VERSION"
        echo "可选: baseline, v2_kvcache, v3_kvcache_noslice, v4_noexpand"
        exit 1
        ;;
esac

ONNX_OUTPUT_DIR="./output/onnx_${MODEL_NAME}_${KV_CACHE_LENGTH}"
ONNX_MODEL_PATH="${ONNX_OUTPUT_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="scripts/logs/1_export_onnx_${MODELING_VERSION}_${TIMESTAMP}.log"
mkdir -p scripts/logs "$ONNX_OUTPUT_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo " [1] ONNX 模型导出" | tee -a "$LOG_FILE"
echo " Modeling: ${MODELING_VERSION} (${MODELING_FILE})" | tee -a "$LOG_FILE"
echo " Model: ${MODEL_NAME}" | tee -a "$LOG_FILE"
echo " KV Cache Length: ${KV_CACHE_LENGTH}" | tee -a "$LOG_FILE"
echo " Output: ${ONNX_MODEL_PATH}" | tee -a "$LOG_FILE"
echo " Time: ${TIMESTAMP}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 替换 modeling 文件（备份原文件）
EXPORT_MODELING="export/modeling_qwen2.py"
if [ "$MODELING_FILE" != "$EXPORT_MODELING" ]; then
    echo ">>> 替换 modeling 文件: ${MODELING_FILE} -> ${EXPORT_MODELING}" | tee -a "$LOG_FILE"
    cp "$EXPORT_MODELING" "${EXPORT_MODELING}.bak"
    cp "$MODELING_FILE" "$EXPORT_MODELING"
fi

# 执行导出
echo ">>> 开始导出 ONNX..." | tee -a "$LOG_FILE"
python3 export/export_onnx.py \
    --device_str "$DEVICE_STR" \
    --dtype "$DTYPE" \
    --hf_model_dir "$HF_MODEL_DIR" \
    --onnx_model_path "$ONNX_MODEL_PATH" \
    --kv_cache_length "$KV_CACHE_LENGTH" \
    --simplify "$SIMPLIFY" \
    2>&1 | tee -a "$LOG_FILE"

# 恢复原文件
if [ -f "${EXPORT_MODELING}.bak" ]; then
    mv "${EXPORT_MODELING}.bak" "$EXPORT_MODELING"
    echo ">>> 已恢复原始 modeling 文件" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 导出完成! ONNX: ${ONNX_MODEL_PATH}" | tee -a "$LOG_FILE"
echo " 日志: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
