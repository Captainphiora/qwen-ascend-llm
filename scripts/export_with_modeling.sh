#!/bin/bash
# ============================================================
# 一站式 ONNX/OM 导出脚本
#
# 功能: export_onnx → change_node → (可选 simplify) → onnx2om
#
# 使用方式:
# bash scripts/export_with_modeling.sh \
#     --modeling_file=export/modeling_qwen2_v4_noexpand.py \
#     --change_node_file=export/change_node_v4_noexpand.py \
    # --simplify
# SOC_VERSION=Ascend310B1
# 输出目录结构 (参考 opt_models/v4_noexpand):
#   output/<version>/
#   ├── onnx_raw/            原始导出的 ONNX
#   ├── onnx_changed/        change_node 处理后的 ONNX
#   ├── onnx_changed_sim/    change_node + simplify 后的 ONNX (仅 --simplify)
#   └── <model>_<kv>_<prefill>_<version>[_sim].om
#
# 参数:
#   --modeling_file=<path>       modeling 文件 (默认: 当前 export/modeling_qwen2.py)
#   --change_node_file=<path>   change_node 脚本 (默认: export/change_node.py)
#   --hf_model_dir=<path>       HuggingFace 模型目录
#   --kv_cache_length=<int>     KV Cache 长度 (默认: 4096)
#   --max_prefill_length=<int>  最大 prefill 长度 (默认: 1)
#   --device_str=<str>          设备类型 (默认: npu)
#   --dtype=<str>               数据类型 (默认: float16)
#   --simplify=<true|false>     对 change_node 后的 ONNX 做 simplify (默认: false)
#   --skip_om                   跳过 OM 编译
#   --soc_version=<str>         NPU 芯片型号 (默认: auto)
#   --cpu_thread=<int>          ATC 编译线程数 (默认: 1)
#   --version=<str>             版本标签，决定输出子目录名 (自动推断)
#   --help/-h                   显示帮助信息
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 默认配置 ----
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
DEVICE_STR="npu"
DTYPE="float16"
# SIMPLIFY=false
SIMPLIFY=true
SKIP_OM=false
SOC_VERSION="auto"
# SOC_VERSION="Ascend310B1"
CPU_THREAD=1
MODELING_FILE=""
# CHANGE_NODE_FILE="export/change_node.py"
CHANGE_NODE_FILE="export/change_node_v4_noexpand_310b.py"
VERSION="no_rope"
# ---- 配置结束 ----

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --modeling_file=*) MODELING_FILE="${arg#*=}" ;;
        --change_node_file=*) CHANGE_NODE_FILE="${arg#*=}" ;;
        --hf_model_dir=*) HF_MODEL_DIR="${arg#*=}" ;;
        --kv_cache_length=*) KV_CACHE_LENGTH="${arg#*=}" ;;
        --max_prefill_length=*) MAX_PREFILL_LENGTH="${arg#*=}" ;;
        --device_str=*) DEVICE_STR="${arg#*=}" ;;
        --dtype=*) DTYPE="${arg#*=}" ;;
        --soc_version=*) SOC_VERSION="${arg#*=}" ;;
        --cpu_thread=*) CPU_THREAD="${arg#*=}" ;;
        --version=*) VERSION="${arg#*=}" ;;
        --simplify=*) SIMPLIFY="${arg#*=}" ;;
        --skip_om) SKIP_OM=true ;;
        --help|-h)
            sed -n '2,36p' "$0"
            exit 0
            ;;
        *)
            echo "[WARN] 未知参数: $arg"
            ;;
    esac
done

# 校验 simplify 参数
if [ "$SIMPLIFY" != "true" ] && [ "$SIMPLIFY" != "false" ]; then
    echo "[ERROR] --simplify 只接受 true 或 false，当前值: $SIMPLIFY"
    exit 1
fi

# ============================================================
# 自动推断版本标签 (从 modeling_file 文件名提取)
# ============================================================
if [ -z "$VERSION" ]; then
    if [ -n "$MODELING_FILE" ]; then
        # e.g. export/modeling_qwen2_v4_noexpand.py -> v4_noexpand
        BASENAME=$(basename "$MODELING_FILE" .py)
        VERSION="${BASENAME#modeling_qwen2_}"
    else
        VERSION="baseline"
    fi
fi

# simplify 后缀
if [ "$SIMPLIFY" = "true" ]; then
    SIM_TAG="_sim"
else
    SIM_TAG=""
fi

# ============================================================
# 输出目录结构
# ============================================================
OUTPUT_BASE="./output/${VERSION}"
ONNX_RAW_DIR="${OUTPUT_BASE}/onnx_raw"
ONNX_CHANGED_DIR="${OUTPUT_BASE}/onnx_changed"
ONNX_CHANGED_SIM_DIR="${OUTPUT_BASE}/onnx_changed_sim"

mkdir -p "$ONNX_RAW_DIR" "$ONNX_CHANGED_DIR"
if [ "$SIMPLIFY" = "true" ]; then
    mkdir -p "$ONNX_CHANGED_SIM_DIR"
fi

ONNX_FILENAME="${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
ONNX_RAW_PATH="${ONNX_RAW_DIR}/${ONNX_FILENAME}"
ONNX_CHANGED_PATH="${ONNX_CHANGED_DIR}/${ONNX_FILENAME}"
ONNX_CHANGED_SIM_PATH="${ONNX_CHANGED_SIM_DIR}/${ONNX_FILENAME}"

OM_NAME="${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}_${VERSION}${SIM_TAG}"
OM_PATH="${OUTPUT_BASE}/${OM_NAME}.om"

# ============================================================
# 验证文件存在
# ============================================================
if [ -n "$MODELING_FILE" ] && [ ! -f "$MODELING_FILE" ]; then
    echo "[ERROR] modeling 文件不存在: $MODELING_FILE"
    exit 1
fi
if [ ! -f "$CHANGE_NODE_FILE" ]; then
    echo "[ERROR] change_node 文件不存在: $CHANGE_NODE_FILE"
    exit 1
fi

# ============================================================
# 打印配置
# ============================================================
echo "============================================================"
echo " 一站式 ONNX/OM 导出"
echo "============================================================"
echo " 版本标签:       ${VERSION}"
echo " Modeling 文件:  ${MODELING_FILE:-export/modeling_qwen2.py (默认)}"
echo " ChangeNode:     ${CHANGE_NODE_FILE}"
echo " Simplify:       ${SIMPLIFY}"
echo " HF 模型目录:   ${HF_MODEL_DIR}"
echo " KV Cache:       ${KV_CACHE_LENGTH}"
echo " Prefill:        ${MAX_PREFILL_LENGTH}"
echo " SOC:            ${SOC_VERSION}"
echo " 输出目录:       ${OUTPUT_BASE}/"
echo "   onnx_raw/       → 原始 ONNX"
echo "   onnx_changed/   → change_node 后"
if [ "$SIMPLIFY" = "true" ]; then
echo "   onnx_changed_sim/ → simplify 后"
fi
echo "   ${OM_NAME}.om"
echo "============================================================"
echo ""

# ============================================================
# Modeling 文件替换逻辑
# ============================================================
EXPORT_MODELING="export/modeling_qwen2.py"
NEED_RESTORE=false

cleanup() {
    if [ "$NEED_RESTORE" = true ] && [ -f "${EXPORT_MODELING}.bak" ]; then
        mv "${EXPORT_MODELING}.bak" "$EXPORT_MODELING"
        echo ">>> 已恢复原始 modeling 文件"
    fi
}
trap cleanup EXIT

if [ -n "$MODELING_FILE" ]; then
    REAL_MODELING=$(realpath "$MODELING_FILE")
    REAL_EXPORT=$(realpath "$EXPORT_MODELING")
    if [ "$REAL_MODELING" != "$REAL_EXPORT" ]; then
        echo ">>> 备份: ${EXPORT_MODELING} -> ${EXPORT_MODELING}.bak"
        cp "$EXPORT_MODELING" "${EXPORT_MODELING}.bak"
        echo ">>> 替换: ${MODELING_FILE} -> ${EXPORT_MODELING}"
        cp "$MODELING_FILE" "$EXPORT_MODELING"
        NEED_RESTORE=true
    fi
fi

# ============================================================
# Step 1: PyTorch → ONNX (export_onnx.py)
# ============================================================
if [ -f "$ONNX_RAW_PATH" ]; then
    echo "[Step 1] 已存在，跳过: $ONNX_RAW_PATH"
else
    echo "[Step 1] PyTorch → ONNX 导出..."
    python3 export/export_onnx.py \
        --device_str "$DEVICE_STR" \
        --dtype "$DTYPE" \
        --hf_model_dir "$HF_MODEL_DIR" \
        --onnx_model_path "$ONNX_RAW_PATH" \
        --kv_cache_length "$KV_CACHE_LENGTH" \
        --simplify false
    echo "[Step 1] 完成: $ONNX_RAW_PATH"
fi
echo ""

# ============================================================
# Step 2: change_node
# ============================================================
if [ -f "$ONNX_CHANGED_PATH" ]; then
    echo "[Step 2] 已存在，跳过: $ONNX_CHANGED_PATH"
else
    echo "[Step 2] 执行 change_node: $(basename $CHANGE_NODE_FILE)"
    python3 "$CHANGE_NODE_FILE" \
        --input_model_path "$ONNX_RAW_PATH" \
        --output_model_path "$ONNX_CHANGED_PATH"
    echo "[Step 2] 完成: $ONNX_CHANGED_PATH"
fi
echo ""

# ============================================================
# Step 3: Simplify (由 --simplify=true/false 控制)
# ============================================================
ONNX_FOR_OM="$ONNX_CHANGED_PATH"

if [ "$SIMPLIFY" = "true" ]; then
    if [ -f "$ONNX_CHANGED_SIM_PATH" ]; then
        echo "[Step 3] 已存在，跳过: $ONNX_CHANGED_SIM_PATH"
    else
        echo "[Step 3] ONNX Simplify..."
        python3 export/simplify_onnx.py \
            --input "$ONNX_CHANGED_PATH" \
            --output "$ONNX_CHANGED_SIM_PATH"
        echo "[Step 3] 完成: $ONNX_CHANGED_SIM_PATH"
    fi
    ONNX_FOR_OM="$ONNX_CHANGED_SIM_PATH"
else
    echo "[Step 3] 跳过 Simplify (--simplify=false)"
fi
echo ""

# ============================================================
# Step 4: ONNX → OM (onnx2om.py)
# ============================================================
if [ "$SKIP_OM" = true ]; then
    echo "[Step 4] 跳过 OM 编译"
else
    echo "[Step 4] ONNX → OM 编译..."
    python3 export/onnx2om.py \
        --hf_model_dir "$HF_MODEL_DIR" \
        --onnx_model_path "$ONNX_FOR_OM" \
        --om_model_path "${OM_PATH%.om}" \
        --kv_cache_length "$KV_CACHE_LENGTH" \
        --max_prefill_length "$MAX_PREFILL_LENGTH" \
        --max_batch 1 \
        --cpu_thread "$CPU_THREAD" \
        --soc_version "$SOC_VERSION"
    echo "[Step 4] 完成: $OM_PATH"
fi
echo ""

# ============================================================
# 完成
# ============================================================
echo "============================================================"
echo " 全部完成！输出目录: ${OUTPUT_BASE}/"
echo "============================================================"
ls -lh "$OUTPUT_BASE"/ 2>/dev/null || true
