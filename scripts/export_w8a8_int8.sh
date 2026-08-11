#!/bin/bash
# ============================================================
# W8A8 INT8 量化模型一键导出脚本
#
# 功能: export_onnx(W8A8) → change_node(AscendQuant+AscendDequant) → onnx2om
#
# 使用方式:
#   bash scripts/export_w8a8_int8.sh
#
# 前置条件:
#   1. 已完成 msmodelslim 量化 (output/quant_w8a8/ 存在)
#   2. source ~/.bashrc 能正确设置 CANN 环境
#
# 输出:
#   output/w8a8_int8/
#   ├── onnx_raw/       export_onnx 导出的原始 ONNX
#   ├── onnx_changed/   change_node 处理后的 ONNX (含 AscendQuant/AscendDequant)
#   └── om/             编译好的 .om 模型
# ============================================================

set -e

# ============================================================
# 配置区 (修改这里的参数即可)
# ============================================================
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
QUANT_MODEL_DIR="./output/quant_w8a8"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
DEVICE_STR="npu"
DTYPE="float16"
SOC_VERSION="auto"
CPU_THREAD=4
PRECISION_MODE="origin"
DEVICE_ID=0
# ============================================================

# 环境初始化
source ~/.bashrc
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# 输出目录
OUTPUT_BASE="./output/w8a8_int8"
ONNX_RAW_DIR="${OUTPUT_BASE}/onnx_raw"
ONNX_CHANGED_DIR="${OUTPUT_BASE}/onnx_changed"
OM_DIR="${OUTPUT_BASE}/om"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "$ONNX_RAW_DIR" "$ONNX_CHANGED_DIR" "$OM_DIR" "$LOG_DIR"

ONNX_FILENAME="${MODEL_NAME}_w8a8_${KV_CACHE_LENGTH}.onnx"
ONNX_RAW_PATH="${ONNX_RAW_DIR}/${ONNX_FILENAME}"
ONNX_CHANGED_PATH="${ONNX_CHANGED_DIR}/${ONNX_FILENAME}"
OM_NAME="${MODEL_NAME}_w8a8_int8_${KV_CACHE_LENGTH}"
OM_PATH="${OM_DIR}/${OM_NAME}"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/export_${TIMESTAMP}.log"

# 打印配置
echo "============================================================"
echo " W8A8 INT8 量化模型导出"
echo "============================================================"
echo " 模型:           ${MODEL_NAME}"
echo " 量化模型目录:   ${QUANT_MODEL_DIR}"
echo " KV Cache:       ${KV_CACHE_LENGTH}"
echo " Prefill:        ${MAX_PREFILL_LENGTH}"
echo " SOC:            ${SOC_VERSION}"
echo " Precision Mode: ${PRECISION_MODE}"
echo " 输出目录:       ${OUTPUT_BASE}/"
echo " 日志:           ${LOG_FILE}"
echo "============================================================"
echo ""

# 全部输出同时写入日志
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================
# Step 1: PyTorch → ONNX (带 W8A8 量化权重导出)
# ============================================================
echo "[Step 1/3] export_onnx.py (W8A8, ${DEVICE_STR}, ${DTYPE})..."
if [ -f "$ONNX_RAW_PATH" ]; then
    echo "  -> 已存在，跳过。删除 ${ONNX_RAW_PATH} 可重新导出。"
else
    python3 export/export_onnx.py \
        --device_str "$DEVICE_STR" \
        --dtype "$DTYPE" \
        --hf_model_dir "$QUANT_MODEL_DIR" \
        --onnx_model_path "$ONNX_RAW_PATH" \
        --kv_cache_length "$KV_CACHE_LENGTH" \
        --simplify false \
        --quantize W8A8
    echo "  -> 完成: $ONNX_RAW_PATH"
fi
echo ""

# ============================================================
# Step 2: change_node (插入 AscendQuant/AscendDequant, 重连 INT8 权重)
# ============================================================
echo "[Step 2/3] change_node.py (AscendQuant + AscendDequant)..."
if [ -f "$ONNX_CHANGED_PATH" ]; then
    echo "  -> 已存在，跳过。删除 ${ONNX_CHANGED_PATH} 可重新执行。"
else
    python3 export/change_node.py \
        --input_model_path "$ONNX_RAW_PATH" \
        --output_model_path "$ONNX_CHANGED_PATH" \
        --quant_model_dir "$QUANT_MODEL_DIR"
    echo "  -> 完成: $ONNX_CHANGED_PATH"
fi
echo ""

# ============================================================
# Step 3: ONNX → OM (ATC 编译)
# ============================================================
echo "[Step 3/3] onnx2om.py (precision_mode=${PRECISION_MODE})..."
if [ -f "${OM_PATH}.om" ]; then
    echo "  -> 已存在，跳过。删除 ${OM_PATH}.om 可重新编译。"
else
    python3 export/onnx2om.py \
        --hf_model_dir "$HF_MODEL_DIR" \
        --onnx_model_path "$ONNX_CHANGED_PATH" \
        --om_model_path "$OM_PATH" \
        --kv_cache_length "$KV_CACHE_LENGTH" \
        --max_prefill_length "$MAX_PREFILL_LENGTH" \
        --max_batch 1 \
        --cpu_thread "$CPU_THREAD" \
        --soc_version "$SOC_VERSION" \
        --precision_mode "$PRECISION_MODE"
    echo "  -> 完成: ${OM_PATH}.om"
fi
echo ""

# ============================================================
# 完成
# ============================================================
echo "============================================================"
echo " 导出完成!"
echo "============================================================"
echo ""
echo " 文件大小:"
ls -lh "${OM_PATH}.om" 2>/dev/null || echo "  (编译失败)"
echo ""
echo " 推理命令:"
echo "   python cli_chat.py --session_type acl \\"
echo "     --om_model_path ${OM_PATH}.om \\"
echo "     --hf_model_dir ${HF_MODEL_DIR} \\"
echo "     --max_output_length ${KV_CACHE_LENGTH}"
echo ""
echo " 日志: ${LOG_FILE}"
echo "============================================================"
