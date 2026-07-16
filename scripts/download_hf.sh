export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download \
  Qwen/Qwen3.6-35B-A3B \
  --local-dir /mnt/host-model/cxj/models/Qwen3.6-35B-A3B \
  --local-dir-use-symlinks False

