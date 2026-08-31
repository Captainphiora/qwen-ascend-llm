#!/bin/bash
# ============================================================
# 脚本1: ONNX + change_node + OM 模型导出
# 完整流程: 导出 ONNX → change_node 图改写 → 编译 OM
#
# 使用方式:
#   bash scripts/1_export_onnx.sh                                    # 默认配置
#   bash scripts/1_export_onnx.sh --modeling=v4_noexpand             # 指定 modeling
#   bash scripts/1_export_onnx.sh --modeling=baseline                # baseline 对比
#   bash scripts/1_export_onnx.sh --max_prefill_length=8             # 设置 prefill 长度
#   bash scripts/1_export_onnx.sh --simplify                         # 启用 onnxsim
#   bash scripts/1_export_onnx.sh --soc=Ascend310B1                  # 310B 编译
#   bash scripts/1_export_onnx.sh --skip-om                          # 仅导出 ONNX，不做 change_node 和 OM
#
# 可用 modeling 版本 (对应 change_node 脚本自动匹配):
#   baseline              - export/modeling_qwen2.py + export/change_node.py
#   v2_kvcache            - export/modeling_qwen2_v2_kvcache.py + export/change_node_v2_kvcache.py
#   v3_kvcache_noslice    - export/modeling_qwen2_v3_kvcache_noslice.py + export/change_node_v3_kvcache_noslice.py
#   v4_noexpand           - export/modeling_qwen2_v4_noexpand.py + export/change_node_v4_noexpand.py
#                           (310B 自动使用 change_node_v4_noexpand_310b.py)
#
# 示例:
#   # 完整流程: v4_noexpand, prefill=1, 910 芯片
#   bash scripts/1_export_onnx.sh --modeling=v4_noexpand --kv_cache_length=4096 --max_prefill_length=1
#
#   # 310B 芯片, prefill=1
#   bash scripts/1_export_onnx.sh --modeling=v4_noexpand --soc=Ascend310B1 --max_prefill_length=1
#
#   # 带 simplify, prefill=8
#   bash scripts/1_export_onnx.sh --modeling=v4_noexpand --max_prefill_length=8 --simplify
#
#   # 仅导出 ONNX (不做 change_node 和 OM 编译)
#   bash scripts/1_export_onnx.sh --modeling=v4_noexpand --skip-om
#
# 输出:
#   ONNX (raw):     ./output/onnx_<MODEL>_<KV_LEN>/<MODEL>_<KV_LEN>.onnx
#   ONNX (changed): ./output/onnx_changed_<modeling>/<MODEL>_<KV_LEN>.onnx
#   OM:             ./output/om_<modeling>/<MODEL>_<KV_LEN>_<PREFILL>[_sim].om
#   日志:           scripts/logs/1_export_<modeling>_<timestamp>.log
#
# OM 文件命名规则:
#   <MODEL>_<kv_cache_length>_<max_prefill_length>[_sim].om
#   例: DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om
#       DeepSeek-R1-Distill-Qwen-1.5B_4096_8_sim.om
# ============================================================

set -e
source ~/.bashrc_cann900

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 默认配置 ----
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/${MODEL_NAME}"
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
DEVICE_ID=7
DEVICE_STR="npu"
DTYPE="float16"
SIMPLIFY="false"
MODELING_VERSION="v4_noexpand"
SOC_VERSION="auto"
CPU_THREAD=64
SKIP_OM=false
# ---- 配置结束 ----

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --modeling=*) MODELING_VERSION="${arg#*=}" ;;
        --kv_cache_length=*) KV_CACHE_LENGTH="${arg#*=}" ;;
        --max_prefill_length=*) MAX_PREFILL_LENGTH="${arg#*=}" ;;
        --device_id=*) DEVICE_ID="${arg#*=}" ;;
        --simplify) SIMPLIFY="true" ;;
        --soc=*) SOC_VERSION="${arg#*=}" ;;
        --skip-om) SKIP_OM=true ;;
        --help|-h)
            sed -n '2,42p' "$0"
            exit 0
            ;;
    esac
done

# 设置 NPU 设备 (导出 ONNX 时需要)
export ASCEND_RT_VISIBLE_DEVICES=$DEVICE_ID

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

# 输出路径
ONNX_OUTPUT_DIR="./output/onnx_${MODEL_NAME}_${KV_CACHE_LENGTH}"
ONNX_MODEL_PATH="${ONNX_OUTPUT_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"

# OM 命名: <model>_<kv>_<prefill>[_sim]
OM_SUFFIX="${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}"
if [ "$SIMPLIFY" = "true" ]; then
    OM_SUFFIX="${OM_SUFFIX}_sim"
fi
OM_OUTPUT_DIR="./output/om_${MODELING_VERSION}"
OM_MODEL_PATH="${OM_OUTPUT_DIR}/${OM_SUFFIX}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="scripts/logs/1_export_${MODELING_VERSION}_${TIMESTAMP}.log"
mkdir -p scripts/logs "$ONNX_OUTPUT_DIR" "$OM_OUTPUT_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo " [1] ONNX + OM 模型导出" | tee -a "$LOG_FILE"
echo " Modeling: ${MODELING_VERSION} (${MODELING_FILE})" | tee -a "$LOG_FILE"
echo " Model: ${MODEL_NAME}" | tee -a "$LOG_FILE"
echo " KV Cache Length: ${KV_CACHE_LENGTH}" | tee -a "$LOG_FILE"
echo " Max Prefill Length: ${MAX_PREFILL_LENGTH}" | tee -a "$LOG_FILE"
echo " Simplify: ${SIMPLIFY}" | tee -a "$LOG_FILE"
echo " SOC: ${SOC_VERSION}" | tee -a "$LOG_FILE"
echo " ONNX Output: ${ONNX_MODEL_PATH}" | tee -a "$LOG_FILE"
if [ "$SKIP_OM" = false ]; then
echo " OM Output: ${OM_MODEL_PATH}.om" | tee -a "$LOG_FILE"
fi
echo " Time: ${TIMESTAMP}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================
# Step 1: 导出 ONNX
# ============================================================
EXPORT_MODELING="export/modeling_qwen2.py"
if [ "$MODELING_FILE" != "$EXPORT_MODELING" ]; then
    echo ">>> 替换 modeling 文件: ${MODELING_FILE} -> ${EXPORT_MODELING}" | tee -a "$LOG_FILE"
    cp "$EXPORT_MODELING" "${EXPORT_MODELING}.bak"
    cp "$MODELING_FILE" "$EXPORT_MODELING"
fi

echo ">>> [Step 1] 导出 ONNX..." | tee -a "$LOG_FILE"
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
echo ">>> ONNX 导出完成: ${ONNX_MODEL_PATH}" | tee -a "$LOG_FILE"

# ============================================================
# Step 2: change_node (ONNX 图改写)
# ============================================================
if [ "$SKIP_OM" = false ]; then
    # 映射 change_node 脚本
    case "$MODELING_VERSION" in
        baseline)           CHANGE_NODE_FILE="export/change_node.py" ;;
        v2_kvcache)         CHANGE_NODE_FILE="export/change_node_v2_kvcache.py" ;;
        v3_kvcache_noslice) CHANGE_NODE_FILE="export/change_node_v3_kvcache_noslice.py" ;;
        v4_noexpand)
            if [ "$SOC_VERSION" = "Ascend310B1" ]; then
                CHANGE_NODE_FILE="export/change_node_v4_noexpand_310b.py"
            else
                CHANGE_NODE_FILE="export/change_node_v4_noexpand.py"
            fi
            ;;
    esac

    ONNX_CHANGED_DIR="./output/onnx_changed_${MODELING_VERSION}"
    ONNX_CHANGED_PATH="${ONNX_CHANGED_DIR}/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
    mkdir -p "$ONNX_CHANGED_DIR"

    echo "" | tee -a "$LOG_FILE"
    echo ">>> [Step 2] change_node 图改写: ${CHANGE_NODE_FILE}" | tee -a "$LOG_FILE"
    python3 "$CHANGE_NODE_FILE" \
        --input_model_path "$ONNX_MODEL_PATH" \
        --output_model_path "$ONNX_CHANGED_PATH" \
        2>&1 | tee -a "$LOG_FILE"

    echo ">>> change_node 完成: ${ONNX_CHANGED_PATH}" | tee -a "$LOG_FILE"

    # ============================================================
    # Step 3: 编译 OM
    # ============================================================
    echo "" | tee -a "$LOG_FILE"
    echo ">>> [Step 3] 编译 OM (max_prefill_length=${MAX_PREFILL_LENGTH})..." | tee -a "$LOG_FILE"

    python3 export/onnx2om.py \
        --onnx_model_path "$ONNX_CHANGED_PATH" \
        --om_model_path "$OM_MODEL_PATH" \
        --hf_model_dir "$HF_MODEL_DIR" \
        --kv_cache_length "$KV_CACHE_LENGTH" \
        --max_prefill_length "$MAX_PREFILL_LENGTH" \
        --soc_version "$SOC_VERSION" \
        --cpu_thread "$CPU_THREAD" \
        2>&1 | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
    echo ">>> OM 编译完成: ${OM_MODEL_PATH}.om" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 导出完成!" | tee -a "$LOG_FILE"
echo " ONNX: ${ONNX_MODEL_PATH}" | tee -a "$LOG_FILE"
if [ "$SKIP_OM" = false ]; then
echo " OM:   ${OM_MODEL_PATH}.om" | tee -a "$LOG_FILE"
fi
echo " 日志: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
