#!/bin/bash
# Benchmark: DeepSeek-R1-Distill-Qwen-1.5B on Ascend 310B
# 对比 512 / 1024 / 2048 输入长度下的推理性能
#
# 输出:
#   - 终端打印格式化表格
#   - result/benchmark_<timestamp>.csv
#   - result/benchmark_<timestamp>.json
#
# 用法:
#   bash scripts/run_benchmark.sh             # 默认参数
#   bash scripts/run_benchmark.sh --verbose   # 显示输入输出内容

# source /usr/local/Ascend/cann-9.0.0/set_env.sh

cd "$(dirname "$0")/.."

# 运行前清理内存
# python cleanup.py --clean 2>/dev/null
  # --input_lengths 512 1024 2048 \
    # --input_lengths 256 512 1024 2048 \

  # --om_model_path "./output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_310b.om" \
# --om_model_path "./output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om" \
HF_MODEL_DIR="../models/DeepSeek-R1-Distill-Qwen-1.5B"
OM_MODEL_PATH="./output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_sim.om"
# 执行 benchmark
python benchmark.py \
  --hf_model_dir $HF_MODEL_DIR \
  --om_model_path $OM_MODEL_PATH \
  --input_lengths 128 256 512 1024 \
  --decode_tokens 256 \
  --num_rounds 1 \
  --num_warmup 0 \
  --output_dir "./result" \
  "$@"
