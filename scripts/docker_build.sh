#!/bin/bash
# ==============================================================================
# Docker 镜像构建脚本
#
# 使用方法：
#   bash scripts/docker_build.sh
#
# 说明：
#   - 镜像包含 Miniconda(msit_compare) + om 模型 + tokenizer
#   - CANN 9.0.0 运行时通过 -v 挂载宿主机目录
#   - 目标机器需安装昇腾 310B 驱动 + CANN 9.0.0
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

IMAGE_NAME="${1:-qwen-ascend-llm}"
IMAGE_TAG="${2:-latest}"

echo "============================================"
echo " Building Docker Image"
echo " Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo " Context: ${PROJECT_DIR}"
echo "============================================"

# ============================================================
# 准备构建上下文
# ============================================================

# 拷贝 Miniconda3（含 msit_compare 环境）
if [ ! -d "${PROJECT_DIR}/miniconda3/envs/msit_compare" ]; then
    echo "[INFO] Copying Miniconda3 + msit_compare env into build context (约4GB)..."
    rm -rf "${PROJECT_DIR}/miniconda3"
    rsync -a --exclude='pkgs/' /usr/local/miniconda3/ "${PROJECT_DIR}/miniconda3/" 2>/dev/null \
        || cp -a /usr/local/miniconda3/ "${PROJECT_DIR}/miniconda3/"
    echo "[INFO] Miniconda3 copy done."
fi

# 清理之前遗留的 cann-9.0.0 拷贝（如有），节省空间
if [ -d "${PROJECT_DIR}/cann-9.0.0" ]; then
    echo "[INFO] Removing old cann-9.0.0 copy from build context (now mounted at runtime)..."
    rm -rf "${PROJECT_DIR}/cann-9.0.0"
fi

# 拷贝 tokenizer 文件（仅需 tokenizer + config，不拷贝 safetensors 权重）
mkdir -p "${PROJECT_DIR}/models/DeepSeek-R1-Distill-Qwen-1.5B"

HF_DIR=""
if [ -d "${PROJECT_DIR}/../models/DeepSeek-R1-Distill-Qwen-1.5B" ]; then
    HF_DIR="$(cd "${PROJECT_DIR}/../models/DeepSeek-R1-Distill-Qwen-1.5B" && pwd)"
elif [ -d "/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B" ]; then
    HF_DIR="/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B"
fi

if [ -z "${HF_DIR}" ]; then
    echo "[ERROR] Cannot find DeepSeek-R1-Distill-Qwen-1.5B model directory"
    exit 1
fi

for f in tokenizer.json tokenizer_config.json config.json generation_config.json; do
    if [ -f "${HF_DIR}/${f}" ]; then
        cp -f "${HF_DIR}/${f}" "${PROJECT_DIR}/models/DeepSeek-R1-Distill-Qwen-1.5B/${f}"
    fi
done
echo "[INFO] Tokenizer files copied."

# 验证关键文件存在
echo "[INFO] Verifying build context..."
for f in "miniconda3/envs/msit_compare/bin/python" \
         "output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_sim.om" \
         "models/DeepSeek-R1-Distill-Qwen-1.5B/tokenizer.json" \
         "models/DeepSeek-R1-Distill-Qwen-1.5B/config.json"; do
    if [ ! -f "${PROJECT_DIR}/${f}" ]; then
        echo "[ERROR] Missing: ${f}"
        exit 1
    fi
done
echo "[INFO] All files verified."

# 构建镜像
echo ""
echo "[INFO] Starting docker build..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo ""
echo "============================================"
echo " Build Complete!"
echo " Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo " Run with:"
echo "   bash scripts/docker_run.sh"
echo "============================================"
