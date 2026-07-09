export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download \
  Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /mnt/host-model/cxj/models/Qwen2.5-0.5B-Instruct \
  --local-dir-use-symlinks False

