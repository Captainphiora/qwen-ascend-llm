#!/bin/bash
# ============================================================
# 快速运行：baseline vs RoPE 优化对比
#
# 用法:
#   bash run_rope_compare.sh
#
# 说明:
#   - v0_baseline: 使用 change_node.py (仅 Trilu/Cast 修复)
#   - v1_rope:     使用 change_node_v1_rope.py (RoPE 融合)
#   两者共享同一份 raw ONNX，区别在于 change_node 步骤
# ============================================================

set -e

WORK_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$WORK_DIR"

ONNX_RAW="output/onnx_DeepSeek-R1-Distill-Qwen-1.5B_4096/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx"

echo "============================================================"
echo " Baseline vs RoPE 优化对比流程"
echo "============================================================"
echo ""

# ---- v0_baseline ----
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo " [1/2] Running v0_baseline"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
bash "$WORK_DIR/run_rope_optimize.sh" \
    --version v0_baseline \
    --change_node change_node.py \
    --skip_export \
    --onnx_input "$ONNX_RAW" \
    --device_id 5 \
    --kv_cache_length 4096 \
    --max_prefill_length 1

echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo " [2/2] Running v1_rope"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
bash "$WORK_DIR/run_rope_optimize.sh" \
    --version v1_rope \
    --change_node change_node_v1_rope.py \
    --skip_export \
    --onnx_input "$ONNX_RAW" \
    --device_id 5 \
    --kv_cache_length 4096 \
    --max_prefill_length 1

echo ""
echo "============================================================"
echo " 对比完成!"
echo "============================================================"
echo ""
echo " 结果对比:"
echo "   Baseline profiling: opt_profiling/v0_baseline/"
echo "   RoPE profiling:     opt_profiling/v1_rope/"
echo "   Baseline benchmark: opt_benchmark/v0_baseline/"
echo "   RoPE benchmark:     opt_benchmark/v1_rope/"
echo ""
echo " 节点对比:"
echo "   Baseline nodes: opt_logs/v0_baseline/node_info_*.txt"
echo "   RoPE nodes:     opt_logs/v1_rope/node_info_*.txt"
echo "============================================================"
