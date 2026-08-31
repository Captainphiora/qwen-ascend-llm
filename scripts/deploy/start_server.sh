#!/bin/bash
# ==============================================================================
# DeepSeek-R1-Distill-Qwen-1.5B API Service Startup Script
#
# Usage:
#   bash scripts/start_server.sh                        # default config
#   bash scripts/start_server.sh --port 8080            # override port
#   bash scripts/start_server.sh --config configs/xxx.json  # custom config
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CONFIG="${PROJECT_DIR}/configs/deepseek_r1_1.5b_310.json"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

if [ ! -f "${CONFIG}" ]; then
    echo "[ERROR] Config not found: ${CONFIG}"
    exit 1
fi

echo "============================================"
echo " Starting LLM API Service"
echo " Config : ${CONFIG}"
echo " Project: ${PROJECT_DIR}"
echo "============================================"

python "${PROJECT_DIR}/server.py" --config "${CONFIG}" ${EXTRA_ARGS}
