#!/bin/bash
# =============================================================================
# Experiment: Compare ONNX simplification impact on inference performance
# Purpose:
#   1. Simplify ONNX model and compare node counts (before vs after)
#   2. Convert simplified ONNX to OM for profiling comparison
#
# File layout (avoids overwriting existing files):
#   Original ONNX:    opt_models/v4_noexpand/onnx_changed/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx
#   Simplified ONNX:  opt_models/v4_noexpand/onnx_changed_sim/DeepSeek-R1-Distill-Qwen-1.5B_4096_sim.onnx
#   Original OM:      output/om_v4_noexpand/DeepSeek-R1-Distill-Qwen-1.5B_4096_8.om
#   Simplified OM:    output/om_v4_noexpand_sim/DeepSeek-R1-Distill-Qwen-1.5B_4096_sim_8.om
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- Config ---
KV_CACHE_LENGTH=4096
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
MAX_PREFILL_LENGTH=8
CPU_THREAD=64

# --- Paths ---
ORIGINAL_ONNX="opt_models/v4_noexpand/onnx_changed/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx"
SIM_ONNX_DIR="opt_models/v4_noexpand/onnx_changed_sim"
SIM_ONNX="${SIM_ONNX_DIR}/DeepSeek-R1-Distill-Qwen-1.5B_4096_sim.onnx"
SIM_OM_PATH="output/om_v4_noexpand_sim/DeepSeek-R1-Distill-Qwen-1.5B_4096_sim_8"

# --- Step 1: Simplify ONNX ---
echo "=========================================="
echo "[Step 1] Simplifying ONNX model..."
echo "  Input:  ${ORIGINAL_ONNX}"
echo "  Output: ${SIM_ONNX}"
echo "=========================================="

mkdir -p "${SIM_ONNX_DIR}"

if [ -f "${SIM_ONNX}" ]; then
    echo "[WARN] Simplified ONNX already exists: ${SIM_ONNX}"
    echo "       Skipping simplification. Delete it to re-run."
else
    python3 export/simplify_onnx.py \
        --input "${ORIGINAL_ONNX}" \
        --output "${SIM_ONNX}"
fi

# --- Step 2: Convert simplified ONNX to OM ---
echo ""
echo "=========================================="
echo "[Step 2] Converting simplified ONNX to OM..."
echo "  Input:  ${SIM_ONNX}"
echo "  Output: ${SIM_OM_PATH}.om"
echo "=========================================="

mkdir -p "$(dirname "${SIM_OM_PATH}")"

if [ -f "${SIM_OM_PATH}.om" ]; then
    echo "[WARN] Simplified OM already exists: ${SIM_OM_PATH}.om"
    echo "       Skipping conversion. Delete it to re-run."
else
    python3 export/onnx2om.py \
        --hf_model_dir="${HF_MODEL_DIR}" \
        --onnx_model_path="${SIM_ONNX}" \
        --om_model_path="${SIM_OM_PATH}" \
        --kv_cache_length=${KV_CACHE_LENGTH} \
        --cpu_thread=${CPU_THREAD} \
        --max_prefill_length=${MAX_PREFILL_LENGTH}
fi

echo ""
echo "=========================================="
echo "[Done] Files for comparison:"
echo "  Original OM:   output/om_v4_noexpand/DeepSeek-R1-Distill-Qwen-1.5B_4096_8.om"
echo "  Simplified OM: ${SIM_OM_PATH}.om"
echo ""
echo "Next steps:"
echo "  - Compare node counts: check onnx_log/ for simplify logs"
echo "  - Profile both OMs and compare operator costs"
echo "=========================================="
