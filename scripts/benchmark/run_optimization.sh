#!/bin/bash
# ============================================================
# 统一导出+编译+测试 脚本
# 根据版本号自动选择对应的 modeling 文件，导出到独立目录
#
# 用法:
#   bash scripts/run_optimization.sh v0_baseline
#   bash scripts/run_optimization.sh v1_rope
#   bash scripts/run_optimization.sh v2_rope_expand
#   bash scripts/run_optimization.sh all        # 顺序跑全部
#
# 优化路线:
#   v0_baseline    - 原始 modeling_qwen2.py，无任何改动
#   v1_rope        - 仅优化 rotate_half (消除 Slice+Neg+Concat)
#   v2_rope_expand - 优化 rotate_half + apply_rotary_pos_emb (消除 Expand)
# ============================================================
set -e
cd "$(dirname "$0")/.."

MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
CPU_THREAD=64

RESULT_DIR="./benchmark_results"
mkdir -p "$RESULT_DIR"

# Benchmark 参数
BENCH_PROMPT="请详细介绍一下机器学习的基本概念和常用算法"
BENCH_MAX_TOKENS=30
BENCH_ROUNDS=3

# ============================================================
# 根据版本选择 modeling 文件
# ============================================================
get_modeling_file() {
    local version=$1
    case $version in
        v0_baseline)    echo "modeling_qwen2.py" ;;
        v1_rope)        echo "modeling_qwen2.py" ;;
        v2_rope_expand) echo "modeling_qwen2_v2_rope_expand.py" ;;
        *) echo ""; return 1 ;;
    esac
}

get_change_node_script() {
    local version=$1
    case $version in
        v1_rope)        echo "change_node_v1_rope.py" ;;
        *)              echo "change_node.py" ;;
    esac
}

# ============================================================
# 单个版本的完整流程: 导出ONNX → change_node → 编译OM → benchmark
# ============================================================
run_version() {
    local VERSION=$1
    local MODELING_FILE=$(get_modeling_file $VERSION)

    if [ -z "$MODELING_FILE" ]; then
        echo "[ERROR] 未知版本: $VERSION"
        echo "可用: v0_baseline, v1_rope, v2_rope_expand"
        return 1
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  版本: ${VERSION}"
    echo "║  Modeling: export/${MODELING_FILE}"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # v0 直接用现有 OM
    if [ "$VERSION" == "v0_baseline" ]; then
        local OM_PATH="./output/model_910_cann900/${MODEL_NAME}_4096_1_sim.om"
        if [ ! -f "$OM_PATH" ]; then
            echo "[ERROR] 基线 OM 不存在: $OM_PATH"
            return 1
        fi
        echo "[INFO] v0 使用现有 OM: $OM_PATH"
        echo ""
        run_benchmark "$VERSION" "$OM_PATH"
        return 0
    fi

    # 输出目录
    local ONNX_DIR="./output/onnx_${VERSION}"
    local ONNX2_DIR="./output/onnx2_${VERSION}"
    local OM_DIR="./output/model_${VERSION}"
    local ONNX_PATH="${ONNX_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
    local ONNX2_PATH="${ONNX2_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}_rectified.onnx"
    local OM_PATH="${OM_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}"

    # Step 1: 设置 modeling 文件（临时替换 export/ 下的 modeling_qwen2.py）
    echo "[Step 1/4] 设置 modeling 文件: ${MODELING_FILE}"
    local EXPORT_DIR="./export"
    local ORIG_MODELING="${EXPORT_DIR}/modeling_qwen2.py"
    local BACKUP_MODELING="${EXPORT_DIR}/_modeling_qwen2_backup.py"

    # 备份原始文件（如果还没备份过）
    if [ ! -f "$BACKUP_MODELING" ]; then
        cp "$ORIG_MODELING" "$BACKUP_MODELING"
    fi
    # 用对应版本替换
    cp "${EXPORT_DIR}/${MODELING_FILE}" "$ORIG_MODELING"

    # Step 2: 导出 ONNX
    echo "[Step 2/4] 导出 ONNX → ${ONNX_DIR}"
    echo "  开始: $(date '+%H:%M:%S')"
    python3 export/export_onnx.py \
        --device_str=npu \
        --dtype=float16 \
        --hf_model_dir="$HF_MODEL_DIR" \
        --onnx_model_path="$ONNX_PATH" \
        --kv_cache_length=$KV_CACHE_LENGTH \
        --simplify=false
    echo "  结束: $(date '+%H:%M:%S')"

    # Step 3: change_node
    local CHANGE_NODE_SCRIPT=$(get_change_node_script $VERSION)
    echo "[Step 3/4] ${CHANGE_NODE_SCRIPT} → ${ONNX2_DIR}"
    python3 export/${CHANGE_NODE_SCRIPT} \
        --input_model_path="$ONNX_PATH" \
        --output_model_path="$ONNX2_PATH"

    # Step 4: 编译 OM
    echo "[Step 4/4] 编译 OM → ${OM_DIR}"
    echo "  开始: $(date '+%H:%M:%S')"
    python3 export/onnx2om.py \
        --hf_model_dir="$HF_MODEL_DIR" \
        --onnx_model_path="$ONNX2_PATH" \
        --om_model_path="$OM_PATH" \
        --kv_cache_length=$KV_CACHE_LENGTH \
        --cpu_thread=$CPU_THREAD \
        --max_prefill_length=$MAX_PREFILL_LENGTH
    echo "  结束: $(date '+%H:%M:%S')"

    # 恢复原始 modeling 文件
    cp "$BACKUP_MODELING" "$ORIG_MODELING"

    # 找到生成的 OM
    local FINAL_OM=$(find "$OM_DIR" -name "*.om" | head -1)
    if [ -z "$FINAL_OM" ]; then
        echo "[ERROR] 编译后未找到 .om 文件"
        return 1
    fi
    echo "[INFO] OM 已生成: $FINAL_OM"

    # Benchmark
    run_benchmark "$VERSION" "$FINAL_OM"
}

# ============================================================
# 运行 benchmark
# ============================================================
run_benchmark() {
    local VERSION=$1
    local OM_PATH=$2
    local RESULT_FILE="${RESULT_DIR}/${VERSION}.txt"

    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo " Benchmark: ${VERSION}"
    echo " OM: ${OM_PATH}"
    echo " 结果: ${RESULT_FILE}"
    echo "──────────────────────────────────────────────────────────────"

    python benchmark.py \
        --om_model_path "$OM_PATH" \
        --hf_model_dir "$HF_MODEL_DIR" \
        --kv_cache_length $KV_CACHE_LENGTH \
        --max_prefill_length $MAX_PREFILL_LENGTH \
        --prompt "$BENCH_PROMPT" \
        --max_new_tokens $BENCH_MAX_TOKENS \
        --rounds $BENCH_ROUNDS \
        --warmup 1 \
        --label "$VERSION" \
        2>&1 | tee "$RESULT_FILE"

    echo ""
    echo "[DONE] ${VERSION} 结果已保存: $RESULT_FILE"
}

# ============================================================
# 主入口
# ============================================================
case "${1:-all}" in
    v0_baseline|v1_rope|v2_rope_expand)
        run_version "$1"
        ;;
    all)
        run_version "v0_baseline"
        run_version "v1_rope"
        run_version "v2_rope_expand"
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║  全部完成! 结果对比:                                        ║"
        echo "╚══════════════════════════════════════════════════════════════╝"
        echo ""
        for f in "$RESULT_DIR"/v*.txt; do
            echo "=== $(basename $f) ==="
            grep -A5 "┌──" "$f" 2>/dev/null || tail -10 "$f"
            echo ""
        done
        ;;
    *)
        echo "用法: bash scripts/run_optimization.sh [VERSION|all]"
        echo ""
        echo "优化路线 (控制变量, 逐个优化):"
        echo "  v0_baseline    - 原始模型, 无优化 (基线)"
        echo "  v1_rope        - 仅优化 rotate_half (消除 Slice+Neg+Concat)"
        echo "  v2_rope_expand - 优化 rotate_half + apply_rotary (消除 Expand)"
        echo "  all            - 顺序执行全部"
        echo ""
        echo "后续可扩展:"
        echo "  v3_xxx         - 新的优化方案..."
        exit 1
        ;;
esac
