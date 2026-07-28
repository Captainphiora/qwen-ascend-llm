#!/bin/bash
# ==============================================================================
# Docker 容器运行脚本
#
# 使用方法：
#   bash scripts/docker_run.sh              # 默认启动服务
#   bash scripts/docker_run.sh bash         # 进入交互式 shell
#
# 环境变量：
#   CANN_PATH  - 宿主机 CANN 安装路径（默认 /usr/local/Ascend/cann-9.0.0）
#   PORT       - 映射端口（默认 8000）
# ==============================================================================
set -e

IMAGE_NAME="${IMAGE_NAME:-qwen-ascend-llm}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PORT="${PORT:-8000}"
CANN_PATH="${CANN_PATH:-/usr/local/Ascend/cann-9.0.0}"

DOCKER_CMD="${@:-bash scripts/start_server.sh --config configs/deepseek_r1_1.5b_310_docker.json}"

if [ ! -d "${CANN_PATH}" ]; then
    echo "[ERROR] CANN not found at: ${CANN_PATH}"
    echo "[ERROR] Please install CANN 9.0.0 or set CANN_PATH env variable."
    exit 1
fi

echo "[INFO] Starting container: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "[INFO] CANN path: ${CANN_PATH}"
echo "[INFO] Port mapping: ${PORT}:8000"
echo "[INFO] Command: ${DOCKER_CMD}"
echo ""

docker run -it --rm \
    -p ${PORT}:8000 \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/svm0 \
    --device=/dev/ts_aisle \
    --device=/dev/upgrade \
    --device=/dev/sys \
    --device=/dev/dvpp_cmdlist \
    --device=/dev/vdec \
    --device=/dev/vpc \
    --device=/dev/pngd \
    --device=/dev/venc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v ${CANN_PATH}:/usr/local/Ascend/cann-9.0.0 \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v /usr/lib64/libdcmi.so:/usr/lib64/libdcmi.so:ro \
    -v /usr/lib64/libaicpu_processer.so:/usr/lib64/libaicpu_processer.so:ro \
    -v /usr/lib64/libaicpu_prof.so:/usr/lib64/libaicpu_prof.so:ro \
    -v /usr/lib64/libaicpu_sharder.so:/usr/lib64/libaicpu_sharder.so:ro \
    -v /usr/lib64/libaicpu_scheduler.so:/usr/lib64/libaicpu_scheduler.so:ro \
    -v /usr/lib64/libadump.so:/usr/lib64/libadump.so:ro \
    -v /usr/lib64/libtsd_eventclient.so:/usr/lib64/libtsd_eventclient.so:ro \
    -v /usr/lib64/libmpi_dvpp_adapter.so:/usr/lib64/libmpi_dvpp_adapter.so:ro \
    -v /usr/lib64/libstackcore.so:/usr/lib64/libstackcore.so:ro \
    -v /usr/lib64/libunified_timer.so:/usr/lib64/libunified_timer.so:ro \
    -v /usr/lib64/aicpu_kernels/:/usr/lib64/aicpu_kernels/:ro \
    -v /usr/lib/aarch64-linux-gnu/libcrypto.so.1.1:/usr/lib/aarch64-linux-gnu/libcrypto.so.1.1:ro \
    -v /usr/lib/aarch64-linux-gnu/libyaml-0.so.2:/usr/lib/aarch64-linux-gnu/libyaml-0.so.2:ro \
    -v /etc/sys_version.conf:/etc/sys_version.conf:ro \
    -v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg:ro \
    -v /etc/slog.conf:/etc/slog.conf:ro \
    -v /var/slogd:/var/slogd:ro \
    -v /var/dmp_daemon:/var/dmp_daemon:ro \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    ${DOCKER_CMD}
