#!/bin/bash
# Docker entrypoint: source CANN 9.0.0 environment and run the command
set -e

# Driver library path (mounted from host)
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/lib64:${LD_LIBRARY_PATH}

# Source CANN environment
if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

# Activate conda environment (msit_compare)
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate msit_compare 2>/dev/null || true

exec "$@"
