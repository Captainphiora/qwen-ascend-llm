source /usr/local/Ascend/driver/bin/setenv.bash
export LD_LIBRARY_PATH=/root/.local/share/uv/python/cpython-3.10.20-linux-aarch64-gnu/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH

source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/8.5.0/atb/set_env.sh