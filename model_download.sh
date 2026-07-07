#!/bin/bash

# ================= 配置区 =================
MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# 指定你想要下载到的本地文件夹
SAVE_DIR="../models/DeepSeek-R1-Distill-Qwen-1.5B"
LOG_DIR="./log"
LOG_FILE="${LOG_DIR}/download_bash.log"

# ================= 初始化 =================
# 如果 log 目录不存在，则自动创建
mkdir -p "$LOG_DIR"

# 核心：设置环境变量，启用 HF 官方的国内加速镜像
export HF_ENDPOINT="https://hf-mirror.com"

echo "开始下载模型 ${MODEL_ID}..."
echo "日志将实时写入: ${LOG_FILE}"

# ================= 执行下载 =================
# --local-dir: 指定下载到哪个物理目录
# --resume-download: 支持断点续传
# 2>&1 | tee: 将标准输出和错误输出合并，既打印到屏幕，又写入日志文件
huggingface-cli download "$MODEL_ID" \
  --local-dir "$SAVE_DIR" \
  --resume-download \
  2>&1 | tee "$LOG_FILE"

echo "脚本执行完毕，请检查 ${LOG_FILE} 确认是否成功。"