#!/bin/bash
# ============================================================
# 批量编译 OM 脚本（FP16, Ascend910）
#
# 三阶段全并行:
#   Phase 1 — 并行 ONNX 导出（沙箱隔离 modeling + 不同 NPU）
#   Phase 2 — 并行 change_node（纯 CPU，秒级）
#   Phase 3 — 并行 ATC 编译（纯 CPU，充分利用多核）
#
# 沙箱原理:
#   export_onnx.py 用 __file__ 定位项目根并 sys.path.insert(0, root)，
#   因此 PYTHONPATH shadow 会被覆盖。解决方案: symlink 整个项目到沙箱，
#   仅替换 export/modeling_qwen2.py 和模型目录的 config.json，
#   从沙箱内运行脚本使 __file__ 指向沙箱。
#
# 用法:
#   bash scripts/build_all_om.sh
#
# 产物目录:
#   opt_models/<version>/
#     ├── onnx_raw/model.onnx
#     ├── onnx_changed/model.onnx
#     └── om_910/model.om
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 全局配置 ----
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="$(realpath ../models/${MODEL_NAME})"
KV_CACHE_LENGTH=4096
SOC_VERSION="Ascend910_9382"
ATC_THREADS=32
DEFAULT_CHANGE_NODE="export/change_node_v1_rope.py"
SANDBOX_ROOT="/tmp/build_om_$$"
# ---- 全局配置结束 ----

# ---- 版本定义 ----
# 格式: VERSION|MODELING_SUFFIX|EXPORT_EXTRA|OM_EXTRA|CHANGE_NODE_OVERRIDE
VERSIONS=(
  "v3|v3_kvcache_noslice|||"
  "v4|v4_noexpand|||"
  "v5|v5_gate_up_fuse|||"
  "v6|v6_transpose_elim|--kv_cache_layout BHSD|--kv_cache_layout BHSD|"
  "v7b|v7b_split_kv|--kv_cache_layout BHSD|--kv_cache_layout BHSD|"
  "v8|v8_qkv_fuse|||"
  "v9|v9_kv_slice|||"
  "v10|v10_gate_up_prefuse|||"
  "v11|v11_kv_inplace|--kv_inplace|--kv_inplace|export/change_node_v11_kv_inplace.py"
)

TOTAL=${#VERSIONS[@]}
LOG_DIR="${SCRIPT_DIR}/logs/build_all_om_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " 批量编译 OM (FP16, ${SOC_VERSION})"
echo " 版本数: ${TOTAL}"
echo " 沙箱: ${SANDBOX_ROOT}/"
echo " 日志: ${LOG_DIR}/"
echo "============================================================"
echo ""

# ============================================================
# 构建沙箱: symlink 整个项目，仅替换 modeling 和 config
# ============================================================
setup_sandbox() {
  local VER=$1 MODELING=$2
  local SB="${SANDBOX_ROOT}/${VER}"

  # 1) symlink 项目根目录下的所有文件/目录
  mkdir -p "$SB"
  for f in "${SCRIPT_DIR}"/*; do
    ln -sf "$f" "${SB}/$(basename $f)"
  done

  # 2) 复制 export/ 的 .py 文件（不能 symlink，Python 会解析 __file__ 为真实路径）
  #    大文件（.onnx, .om, kernel_meta 等）保持 symlink
  rm -f "${SB}/export"
  mkdir -p "${SB}/export"
  for f in "${SCRIPT_DIR}/export"/*; do
    case "$f" in
      *.py) cp "$f" "${SB}/export/$(basename $f)" ;;
      *)    ln -sf "$f" "${SB}/export/$(basename $f)" ;;
    esac
  done
  cp "${SCRIPT_DIR}/export/modeling_qwen2_${MODELING}.py" "${SB}/export/modeling_qwen2.py"

  # 3) 替换模型目录（symlink 大文件，独立 config.json + modeling）
  rm -f "${SB}/models" 2>/dev/null
  local MODEL_SB="${SB}/models/${MODEL_NAME}"
  mkdir -p "${MODEL_SB}"
  for f in "${HF_MODEL_DIR}"/*; do
    ln -sf "$f" "${MODEL_SB}/$(basename $f)"
  done
  rm -f "${MODEL_SB}/config.json" "${MODEL_SB}/modeling_qwen2.py"
  cp "${HF_MODEL_DIR}/config.json" "${MODEL_SB}/config.json"
  cp "${SCRIPT_DIR}/export/modeling_qwen2_${MODELING}.py" "${MODEL_SB}/modeling_qwen2.py"

  # 4) opt_models 指向真实目录（产物写到项目里，不留在沙箱）
  rm -f "${SB}/opt_models"
  ln -sf "${SCRIPT_DIR}/opt_models" "${SB}/opt_models"
}

# ============================================================
# Phase 1: 并行 ONNX 导出
# ============================================================
echo "==== Phase 1: ONNX 导出 (${TOTAL} 个并行, 沙箱隔离) ===="
PHASE1_START=$(date +%s)
PIDS=()

for i in "${!VERSIONS[@]}"; do
  IFS='|' read -r VER MODELING EXPORT_EXTRA OM_EXTRA CN_OVERRIDE <<< "${VERSIONS[$i]}"
  DIR="opt_models/${VER}"
  NPU_ID=$((i % 16))
  SB="${SANDBOX_ROOT}/${VER}"

  mkdir -p "${DIR}/onnx_raw" "${DIR}/onnx_changed" "${DIR}/om_910"
  setup_sandbox "$VER" "$MODELING"

  echo "  启动 ${VER} (NPU:${NPU_ID}, modeling=${MODELING})..."
  (
    cd "$SB"
    ASCEND_RT_VISIBLE_DEVICES=$NPU_ID \
    python3 export/export_onnx.py \
      --hf_model_dir="${SB}/models/${MODEL_NAME}" \
      --onnx_model_path="${DIR}/onnx_raw/model.onnx" \
      --kv_cache_length="$KV_CACHE_LENGTH" \
      --kv_cache_layout=BSHD \
      --device_str=npu \
      --dtype=float16 \
      ${EXPORT_EXTRA}
  ) > "${LOG_DIR}/${VER}_export.log" 2>&1 &

  PIDS+=($!)
done

echo ""
echo "  等待 ${#PIDS[@]} 个 ONNX 导出完成..."
EXPORT_FAILED=0
for i in "${!PIDS[@]}"; do
  IFS='|' read -r VER _ _ _ _ <<< "${VERSIONS[$i]}"
  if wait "${PIDS[$i]}"; then
    echo "  ✓ ${VER}"
  else
    echo "  ✗ ${VER} (查看 ${LOG_DIR}/${VER}_export.log)"
    EXPORT_FAILED=$((EXPORT_FAILED + 1))
  fi
done

PHASE1_END=$(date +%s)
echo "Phase 1 完成: $(( PHASE1_END - PHASE1_START ))s, 失败: ${EXPORT_FAILED}"
echo ""

# ============================================================
# Phase 2: 并行 change_node
# ============================================================
echo "==== Phase 2: change_node (并行) ===="
PHASE2_START=$(date +%s)
PIDS=()
CN_VERSIONS=()

for i in "${!VERSIONS[@]}"; do
  IFS='|' read -r VER MODELING EXPORT_EXTRA OM_EXTRA CN_OVERRIDE <<< "${VERSIONS[$i]}"
  CN="${CN_OVERRIDE:-$DEFAULT_CHANGE_NODE}"
  DIR="opt_models/${VER}"

  [ ! -f "${DIR}/onnx_raw/model.onnx" ] && echo "  跳过 ${VER} (ONNX 不存在)" && continue

  echo "  启动 ${VER} ($(basename $CN))..."
  python3 "$CN" \
    --input_model_path="${DIR}/onnx_raw/model.onnx" \
    --output_model_path="${DIR}/onnx_changed/model.onnx" \
    > "${LOG_DIR}/${VER}_change_node.log" 2>&1 &
  PIDS+=($!)
  CN_VERSIONS+=("$VER")
done

echo ""
echo "  等待 ${#PIDS[@]} 个 change_node 完成..."
CN_FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  ✓ ${CN_VERSIONS[$i]}"
  else
    echo "  ✗ ${CN_VERSIONS[$i]} (查看 ${LOG_DIR}/${CN_VERSIONS[$i]}_change_node.log)"
    CN_FAILED=$((CN_FAILED + 1))
  fi
done

PHASE2_END=$(date +%s)
echo "Phase 2 完成: $(( PHASE2_END - PHASE2_START ))s"
echo ""

# ============================================================
# Phase 3: 并行 ATC 编译
# ============================================================
echo "==== Phase 3: ATC 编译 (并行, ${ATC_THREADS} 线程/版本) ===="
PHASE3_START=$(date +%s)
PIDS=()
ATC_VERSIONS=()

for i in "${!VERSIONS[@]}"; do
  IFS='|' read -r VER MODELING EXPORT_EXTRA OM_EXTRA CN_OVERRIDE <<< "${VERSIONS[$i]}"
  DIR="opt_models/${VER}"

  [ ! -f "${DIR}/onnx_changed/model.onnx" ] && echo "  跳过 ${VER} (changed ONNX 不存在)" && continue

  echo "  启动 ATC: ${VER}..."
  python3 export/onnx2om.py \
    --hf_model_dir="$HF_MODEL_DIR" \
    --onnx_model_path="${DIR}/onnx_changed/model.onnx" \
    --om_model_path="${DIR}/om_910/model" \
    --kv_cache_length="$KV_CACHE_LENGTH" \
    --kv_cache_layout=BSHD \
    --max_prefill_length=1 \
    --soc_version="$SOC_VERSION" \
    --cpu_thread="$ATC_THREADS" \
    ${OM_EXTRA} \
    > "${LOG_DIR}/${VER}_atc.log" 2>&1 &

  PIDS+=($!)
  ATC_VERSIONS+=("$VER")
done

echo ""
echo "  等待 ${#PIDS[@]} 个 ATC 编译完成..."
ATC_FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  ✓ ${ATC_VERSIONS[$i]}"
  else
    echo "  ✗ ${ATC_VERSIONS[$i]} (查看 ${LOG_DIR}/${ATC_VERSIONS[$i]}_atc.log)"
    ATC_FAILED=$((ATC_FAILED + 1))
  fi
done

PHASE3_END=$(date +%s)
echo "Phase 3 完成: $(( PHASE3_END - PHASE3_START ))s"
echo ""

# ============================================================
# 清理沙箱
# ============================================================
rm -rf "$SANDBOX_ROOT"

# ============================================================
# 汇总
# ============================================================
echo "============================================================"
echo " 编译完成"
echo " Phase 1 (ONNX 并行导出):  $(( PHASE1_END - PHASE1_START ))s"
echo " Phase 2 (change_node):     $(( PHASE2_END - PHASE2_START ))s"
echo " Phase 3 (ATC 并行编译):    $(( PHASE3_END - PHASE3_START ))s"
echo " 总耗时: $(( PHASE3_END - PHASE1_START ))s"
echo " 失败: export=${EXPORT_FAILED}, change_node=${CN_FAILED}, atc=${ATC_FAILED}"
echo ""
echo " 产物:"
for entry in "${VERSIONS[@]}"; do
  IFS='|' read -r VER _ _ _ _ <<< "$entry"
  OM="opt_models/${VER}/om_910/model.om"
  if [ -f "$OM" ]; then
    SIZE=$(du -h "$OM" | cut -f1)
    echo "   ✓ ${OM}  (${SIZE})"
  else
    echo "   ✗ ${OM}  (不存在)"
  fi
done
echo " 日志: ${LOG_DIR}/"
echo "============================================================"
