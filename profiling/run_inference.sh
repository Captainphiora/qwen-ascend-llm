#!/bin/bash
# ============================================================
# Profiling 推理入口 (配置写在脚本顶部, 直接运行即可)
#
# 用法:
#   bash profiling/run_inference.sh
# ============================================================

set -e

# ============================================================
# 【参数配置区】修改这里, 然后直接运行
# ============================================================

# 模型路径
HF_MODEL_DIR="../models/DeepSeek-R1-Distill-Qwen-1.5B"
OM_MODEL_PATH="output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om"

# 设备
DEVICE_ID=0

# 推理参数
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
MAX_NEW_TOKENS=50

# 采样
TEMPERATURE=0.6
SAMPLING_METHOD="top_p"
SAMPLING_VALUE=0.95
USE_NPU_SAMPLING=0               # 0=CPU采样, 1=NPU采样 (需要 torch_npu 且 session_type=acl)

# Prompt
PROMPT="请详细介绍一下机器学习的基本概念和常用算法"

# ============================================================
# 【执行逻辑】
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CANN_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}"
if [ -f "${CANN_HOME}/set_env.sh" ]; then
    source "${CANN_HOME}/set_env.sh" 2>/dev/null || true
fi

export USE_NPU_SAMPLING="$USE_NPU_SAMPLING"

python3 profiling/inference.py \
    --hf_model_dir "$HF_MODEL_DIR" \
    --om_model_path "$OM_MODEL_PATH" \
    --device_id "$DEVICE_ID" \
    --kv_cache_length "$KV_CACHE_LENGTH" \
    --max_prefill_length "$MAX_PREFILL_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --sampling_method "$SAMPLING_METHOD" \
    --sampling_value "$SAMPLING_VALUE" \
    --prompt "$PROMPT"
