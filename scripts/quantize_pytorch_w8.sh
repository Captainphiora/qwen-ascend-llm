#!/bin/bash
# 方案C: PyTorch侧W8A8量化 → ONNX导出 → change_node → atc编译
# 在PyTorch模型中将nn.Linear替换为量化版本(INT8权重+INT8激活)
# 导出ONNX后, Cast→INT8 节点被change_node转为AscendQuant算子
# atc编译时识别量化图, 在NPU上执行INT8 MatMul

set -e
source ~/.bashrc_cann900

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="qwen_ascend_cann900"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=8
CPU_THREAD=64
QUANTIZE_MODE="W8X8"  # W8X8 or W8A16

HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
ONNX_OUTPUT_DIR="output/onnx_${MODEL_NAME}_${KV_CACHE_LENGTH}_${QUANTIZE_MODE}"
ONNX_MODEL_PATH="${ONNX_OUTPUT_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
ONNX_CHANGED_DIR="output/onnx_changed_${QUANTIZE_MODE}"
ONNX_CHANGED_PATH="${ONNX_CHANGED_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
OM_MODEL_PATH="output/om_v4_noexpand_${QUANTIZE_MODE}/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}"

mkdir -p "$ONNX_OUTPUT_DIR" "$ONNX_CHANGED_DIR"

echo "============================================"
echo " Step 1: Export quantized ONNX (${QUANTIZE_MODE})"
echo "============================================"

# 使用v4_noexpand modeling
EXPORT_MODELING="export/modeling_qwen2.py"
cp "$EXPORT_MODELING" "${EXPORT_MODELING}.bak"
cp "export/modeling_qwen2_v4_noexpand.py" "$EXPORT_MODELING"

python3 export/export_onnx.py \
    --device_str npu \
    --dtype float16 \
    --hf_model_dir "$HF_MODEL_DIR" \
    --onnx_model_path "$ONNX_MODEL_PATH" \
    --kv_cache_length "$KV_CACHE_LENGTH" \
    --quantize "$QUANTIZE_MODE"

# 恢复modeling文件
mv "${EXPORT_MODELING}.bak" "$EXPORT_MODELING"

echo ""
echo "============================================"
echo " Step 2: change_node (RoPE fusion + AscendQuant)"
echo "============================================"

python3 export/change_node_v4_noexpand.py \
    --input_model_path "$ONNX_MODEL_PATH" \
    --output_model_path "$ONNX_CHANGED_PATH"

echo ""
echo "============================================"
echo " Step 3: Compile to OM"
echo "============================================"

python3 export/onnx2om.py \
    --hf_model_dir="$HF_MODEL_DIR" \
    --onnx_model_path="$ONNX_CHANGED_PATH" \
    --om_model_path="$OM_MODEL_PATH" \
    --kv_cache_length=$KV_CACHE_LENGTH \
    --cpu_thread=$CPU_THREAD \
    --max_prefill_length=$MAX_PREFILL_LENGTH

echo ""
echo "============================================"
echo " Done!"
echo "============================================"
echo "  Quantize mode: ${QUANTIZE_MODE}"
echo "  ONNX (raw):    ${ONNX_MODEL_PATH}"
echo "  ONNX (changed): ${ONNX_CHANGED_PATH}"
echo "  OM model:      ${OM_MODEL_PATH}.om"
echo ""
echo "Next: benchmark against FP16 baseline"
echo "  FP16 OM: output/om_v4_noexpand/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}.om"
