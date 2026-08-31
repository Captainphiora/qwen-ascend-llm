#!/bin/bash
# ============================================================
# RoPE 优化导出与评估一键脚本
#
# 流程: PyTorch → ONNX → change_node (baseline/rope) → OM → Profiling → Benchmark
#
# 用法:
#   bash run_rope_optimize.sh [OPTIONS]
#
# 选项:
#   --version        版本标识 (如 v0_baseline, v1_rope)
#   --change_node    change_node 脚本 (baseline=change_node.py, rope=change_node_v1_rope.py)
#   --skip_export    跳过 PyTorch→ONNX 导出 (复用已有 ONNX)
#   --skip_om        跳过 ONNX→OM 编译
#   --skip_profiling 跳过 profiling 采集
#   --skip_benchmark 跳过 benchmark
#   --onnx_input     直接指定已有 ONNX 路径 (配合 --skip_export 使用)
#   --device_id      NPU 设备 ID (默认: 5)
#   --kv_cache_length  KV Cache 长度 (默认: 4096)
#   --max_prefill_length  最大 prefill 长度 (默认: 1)
#   --max_new_tokens  profiling 生成 token 数 (默认: 20)
#   --benchmark_tokens  benchmark 生成 token 数 (默认: 30)
#   --benchmark_rounds  benchmark 轮数 (默认: 3)
# ============================================================

set -eo pipefail

# ============================================================
# 默认参数
# ============================================================
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR="/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
VERSION=""
CHANGE_NODE_SCRIPT="change_node.py"
DEVICE_ID=5
KV_CACHE_LENGTH=4096
MAX_PREFILL_LENGTH=1
MAX_NEW_TOKENS=20
BENCHMARK_TOKENS=30
BENCHMARK_ROUNDS=3
CONDA_ENV="qwen_ascend_cann900"

SKIP_EXPORT=false
SKIP_OM=false
SKIP_PROFILING=false
SKIP_BENCHMARK=false
ONNX_INPUT=""
MODELING_FILE="modeling_qwen2.py"

# ============================================================
# 解析命令行参数
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --version) VERSION="$2"; shift 2 ;;
        --change_node) CHANGE_NODE_SCRIPT="$2"; shift 2 ;;
        --skip_export) SKIP_EXPORT=true; shift ;;
        --skip_om) SKIP_OM=true; shift ;;
        --skip_profiling) SKIP_PROFILING=true; shift ;;
        --skip_benchmark) SKIP_BENCHMARK=true; shift ;;
        --onnx_input) ONNX_INPUT="$2"; shift 2 ;;
        --device_id) DEVICE_ID="$2"; shift 2 ;;
        --kv_cache_length) KV_CACHE_LENGTH="$2"; shift 2 ;;
        --max_prefill_length) MAX_PREFILL_LENGTH="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --benchmark_tokens) BENCHMARK_TOKENS="$2"; shift 2 ;;
        --benchmark_rounds) BENCHMARK_ROUNDS="$2"; shift 2 ;;
        --hf_model_dir) HF_MODEL_DIR="$2"; shift 2 ;;
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --conda_env) CONDA_ENV="$2"; shift 2 ;;
        --modeling_file) MODELING_FILE="$2"; shift 2 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "$VERSION" ]; then
    echo "[ERROR] 必须指定 --version (如 v0_baseline, v1_rope)"
    exit 1
fi

# ============================================================
# 路径约定
# ============================================================
WORK_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$WORK_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TAG="${MODEL_NAME}_${KV_CACHE_LENGTH}_${MAX_PREFILL_LENGTH}_${VERSION}"

# 四个独立文件夹
DIR_MODELS="$WORK_DIR/opt_models"
DIR_PROFILING="$WORK_DIR/opt_profiling"
DIR_BENCHMARK="$WORK_DIR/opt_benchmark"
DIR_LOGS="$WORK_DIR/opt_logs"

mkdir -p "$DIR_MODELS/$VERSION"
mkdir -p "$DIR_PROFILING/$VERSION"
mkdir -p "$DIR_BENCHMARK/$VERSION"
mkdir -p "$DIR_LOGS/$VERSION"

# 中间产物路径
ONNX_RAW_DIR="$DIR_MODELS/$VERSION/onnx_raw"
ONNX_CHANGED_DIR="$DIR_MODELS/$VERSION/onnx_changed"
OM_PATH="$DIR_MODELS/$VERSION/${TAG}.om"

mkdir -p "$ONNX_RAW_DIR"
mkdir -p "$ONNX_CHANGED_DIR"

ONNX_RAW_PATH="$ONNX_RAW_DIR/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"
ONNX_CHANGED_PATH="$ONNX_CHANGED_DIR/${MODEL_NAME}_${KV_CACHE_LENGTH}.onnx"

echo "============================================================"
echo " RoPE 优化导出与评估"
echo "============================================================"
echo " 版本:          $VERSION"
echo " 模型:          $MODEL_NAME"
echo " KV Cache:      $KV_CACHE_LENGTH"
echo " Prefill:       $MAX_PREFILL_LENGTH"
echo " 设备:          Device $DEVICE_ID"
echo " change_node:   $CHANGE_NODE_SCRIPT"
echo " 输出目录:      opt_models/$VERSION"
echo " 时间戳:        $TIMESTAMP"
echo "============================================================"
echo ""

# ============================================================
# 环境激活
# ============================================================
echo "[Env] 激活环境..."
set +e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
source ~/.bashrc_cann900
set -eo pipefail
echo ""

# ============================================================
# Step 1: PyTorch → ONNX
# ============================================================
if [ "$SKIP_EXPORT" = true ]; then
    echo "[Step 1] 跳过 ONNX 导出"
    if [ -n "$ONNX_INPUT" ]; then
        ONNX_RAW_PATH="$ONNX_INPUT"
        echo "[Step 1] 使用已有 ONNX: $ONNX_RAW_PATH"
    fi
else
    echo "[Step 1] PyTorch → ONNX 导出..."
    echo "[Step 1] 使用 modeling 文件: $MODELING_FILE"
    cp "$WORK_DIR/export/$MODELING_FILE" "$WORK_DIR/export/modeling_qwen2.py"
    python "$WORK_DIR/export/export_onnx.py" \
        --hf_model_dir "$HF_MODEL_DIR" \
        --onnx_model_path "$ONNX_RAW_PATH" \
        --kv_cache_length "$KV_CACHE_LENGTH" \
        --device_str npu \
        --dtype float16 \
        --simplify false \
        2>&1 | tee "$DIR_LOGS/$VERSION/export_onnx_${TIMESTAMP}.log"
    echo "[Step 1] ONNX 导出完成: $ONNX_RAW_PATH"
fi
echo ""

# ============================================================
# Step 2: change_node (baseline 或 RoPE 融合)
# ============================================================
echo "[Step 2] 执行 change_node: $CHANGE_NODE_SCRIPT"
python "$WORK_DIR/export/$CHANGE_NODE_SCRIPT" \
    --input_model_path "$ONNX_RAW_PATH" \
    --output_model_path "$ONNX_CHANGED_PATH" \
    2>&1 | tee "$DIR_LOGS/$VERSION/change_node_${TIMESTAMP}.log"

# 记录节点信息
python -c "
import onnx
model = onnx.load('$ONNX_CHANGED_PATH')
op_counts = {}
for node in model.graph.node:
    op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
total = len(model.graph.node)
unique = len(op_counts)
print(f'Total nodes: {total}')
print(f'Unique op types: {unique}')
print()
for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
    print(f'  {op:<35} {cnt:>5}')
" | tee "$DIR_LOGS/$VERSION/node_info_${TIMESTAMP}.txt"

echo "[Step 2] change_node 完成: $ONNX_CHANGED_PATH"
echo ""

# ============================================================
# Step 3: ONNX → OM
# ============================================================
if [ "$SKIP_OM" = true ]; then
    echo "[Step 3] 跳过 OM 编译"
    # 需要用户提供已有 OM 或从默认路径查找
    if [ ! -f "$OM_PATH" ]; then
        echo "[ERROR] OM 文件不存在: $OM_PATH"
        exit 1
    fi
else
    echo "[Step 3] ONNX → OM 编译..."
    python "$WORK_DIR/export/onnx2om.py" \
        --hf_model_dir "$HF_MODEL_DIR" \
        --onnx_model_path "$ONNX_CHANGED_PATH" \
        --om_model_path "${OM_PATH%.om}" \
        --kv_cache_length "$KV_CACHE_LENGTH" \
        --max_prefill_length "$MAX_PREFILL_LENGTH" \
        --max_batch 1 \
        2>&1 | tee "$DIR_LOGS/$VERSION/onnx2om_${TIMESTAMP}.log"
    echo "[Step 3] OM 编译完成: $OM_PATH"
fi
echo ""

# ============================================================
# Step 4: Profiling
# ============================================================
if [ "$SKIP_PROFILING" = true ]; then
    echo "[Step 4] 跳过 Profiling"
else
    echo "[Step 4] Profiling 采集..."
    PROF_RAW_DIR="$DIR_PROFILING/$VERSION/raw_${TIMESTAMP}"
    PROF_ANALYSIS="$DIR_PROFILING/$VERSION/analysis_${TAG}_${TIMESTAMP}.txt"
    mkdir -p "$PROF_RAW_DIR"

    python "$WORK_DIR/profiling_collect.py" \
        --om_model_path "$OM_PATH" \
        --hf_model_dir "$HF_MODEL_DIR" \
        --output_dir "$PROF_RAW_DIR" \
        --device_id "$DEVICE_ID" \
        --kv_cache_length "$KV_CACHE_LENGTH" \
        --max_prefill_length "$MAX_PREFILL_LENGTH" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        2>&1 | tee "$DIR_LOGS/$VERSION/profiling_collect_${TIMESTAMP}.log"

    echo "[Step 4] 解析 Profiling 数据..."
    MSPROF=$ASCEND_HOME_PATH/tools/profiler/bin/msprof
    PROF_DIR=$(find "$PROF_RAW_DIR" -maxdepth 2 -name "PROF_*" -type d | sort | tail -n 1)
    if [ -n "$PROF_DIR" ]; then
        $MSPROF --export=on --output="$PROF_DIR" --type=text --summary-format=csv \
            2>&1 | tee -a "$DIR_LOGS/$VERSION/profiling_parse_${TIMESTAMP}.log"

        python "$WORK_DIR/profiling_analyze.py" \
            --prof_dir "$PROF_DIR" \
            --output_file "$PROF_ANALYSIS" \
            --model_name "$TAG" \
            --device_id "$DEVICE_ID" \
            2>&1 | tee -a "$DIR_LOGS/$VERSION/profiling_analyze_${TIMESTAMP}.log"
    else
        echo "[WARN] 未找到 PROF_* 目录"
    fi
    echo "[Step 4] Profiling 完成"
fi
echo ""

# ============================================================
# Step 5: Benchmark
# ============================================================
if [ "$SKIP_BENCHMARK" = true ]; then
    echo "[Step 5] 跳过 Benchmark"
else
    echo "[Step 5] 性能 Benchmark..."
    BENCH_FILE="$DIR_BENCHMARK/$VERSION/benchmark_${TAG}_${TIMESTAMP}.txt"

    python "$WORK_DIR/benchmark.py" \
        --om_model_path "$OM_PATH" \
        --hf_model_dir "$HF_MODEL_DIR" \
        --kv_cache_length "$KV_CACHE_LENGTH" \
        --max_prefill_length "$MAX_PREFILL_LENGTH" \
        --max_new_tokens "$BENCHMARK_TOKENS" \
        --rounds "$BENCHMARK_ROUNDS" \
        --warmup 1 \
        --device_id "$DEVICE_ID" \
        --label "$VERSION" \
        2>&1 | tee "$BENCH_FILE"

    echo "[Step 5] Benchmark 完成: $BENCH_FILE"
fi
echo ""

# ============================================================
# 汇总
# ============================================================
echo "============================================================"
echo " 全部完成! 版本: $VERSION"
echo "============================================================"
echo ""
echo " 模型文件:"
echo "   ONNX (raw):     $ONNX_RAW_PATH"
echo "   ONNX (changed): $ONNX_CHANGED_PATH"
echo "   OM:             $OM_PATH"
echo ""
echo " 结果文件:"
echo "   节点信息:  $DIR_LOGS/$VERSION/node_info_${TIMESTAMP}.txt"
[ "$SKIP_PROFILING" = false ] && echo "   Profiling: $PROF_ANALYSIS"
[ "$SKIP_BENCHMARK" = false ] && echo "   Benchmark: $BENCH_FILE"
echo ""
echo " 日志目录:  $DIR_LOGS/$VERSION/"
echo "============================================================"
