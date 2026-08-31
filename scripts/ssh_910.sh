#!/bin/bash
# ============================================================
# 连接 910 服务器的便捷脚本
#
# 用法:
#   bash scripts/ssh_910.sh                  # 交互式登录
#   bash scripts/ssh_910.sh "npu-smi info"   # 执行单条命令
#   bash scripts/ssh_910.sh < local_script.sh  # 执行本地脚本
#
# 环境:
#   910 服务器: 10.1.30.201:31222 (root)
#   NPU: Ascend910 × 2+, HBM 65536 MB/卡
#   个人目录: /mnt/host-model/cxj (~)
#   项目目录: ~/qwen-ascend-llm
# ============================================================

SSH_HOST="luoss-demo-env"

if [ $# -eq 0 ]; then
    ssh -t "$SSH_HOST" "cd ~/qwen-ascend-llm && exec bash -l"
else
    ssh "$SSH_HOST" "cd ~/qwen-ascend-llm && $*"
fi
