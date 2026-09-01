#!/bin/bash
# 310B1 一键 benchmark: v6_transpose_elim (BHSD layout)
# 用法: bash scripts/bench_v6_transpose_elim.sh [--om_model_path /path/to/v6.om]
#
# 默认 OM 路径: opt_models/v6_transpose_elim_310b/*.om
# 注意: v6 使用 BHSD KV cache 布局

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

DEFAULT_OM="opt_models/v6_transpose_elim_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v6_transpose_elim_310b.om"
HF_MODEL_DIR="../models/DeepSeek-R1-Distill-Qwen-1.5B"

python benchmarks/benchmark.py \
    --om_model_path "${1:-$DEFAULT_OM}" \
    --hf_model_dir "$HF_MODEL_DIR" \
    --kv_cache_length 4096 \
    --max_prefill_length 1 \
    --max_new_tokens 30 \
    --rounds 3 \
    --warmup 1 \
    --kv_cache_layout BHSD \
    --label v6_transpose_elim_310b \
    "$@"
