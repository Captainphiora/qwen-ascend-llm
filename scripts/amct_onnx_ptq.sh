#!/bin/bash
# ============================================================
# AMCT ONNX PTQ 量化脚本
# 使用 amct_onnx 对 FP16 模型做 W8A8 校准量化，产出部署级 ONNX
# 然后用 ATC 编译为 OM
#
# 用法:
#   bash scripts/amct_onnx_ptq.sh [--npu_id=2] [--num_samples=1]
#
# 输出:
#   output/amct_onnx_ptq/model_deploy_deploy_model.onnx  (量化 ONNX)
#   output/amct_onnx_ptq/model_deploy_fake_quant_model.onnx (fake quant ONNX)
#   output/om_ptq_910/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_ptq.om (量化 OM)
# ============================================================

set -e
source ~/.bashrc_cann900
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen_ascend_cann900

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- 默认配置 ----
NPU_ID="3"
NUM_SAMPLES=1
KV_CACHE_LENGTH=4096
MODEL_PATH="output/onnx_changed_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx"
HF_MODEL_DIR="/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR="output/amct_onnx_ptq"
OM_OUTPUT="output/om_ptq_910/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_ptq"
# ---- 配置结束 ----

for arg in "$@"; do
    case "$arg" in
        --npu_id=*) NPU_ID="${arg#*=}" ;;
        --num_samples=*) NUM_SAMPLES="${arg#*=}" ;;
        --kv_cache_length=*) KV_CACHE_LENGTH="${arg#*=}" ;;
    esac
done

export ASCEND_RT_VISIBLE_DEVICES="$NPU_ID"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="scripts/logs/amct_onnx_ptq_${TIMESTAMP}.log"
mkdir -p scripts/logs "$OUTPUT_DIR"

echo "============================================================" | tee "$LOG_FILE"
echo " AMCT ONNX PTQ 量化" | tee -a "$LOG_FILE"
echo " NPU: $NPU_ID" | tee -a "$LOG_FILE"
echo " Samples: $NUM_SAMPLES" | tee -a "$LOG_FILE"
echo " KV Cache: $KV_CACHE_LENGTH" | tee -a "$LOG_FILE"
echo " Model: $MODEL_PATH" | tee -a "$LOG_FILE"
echo " Time: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 1-4: Python 量化流程
echo ">>> 开始 AMCT 量化流程..." | tee -a "$LOG_FILE"
python3 << 'PYTHON_EOF' 2>&1 | tee -a "$LOG_FILE"
import os
import sys
import json
import time
import numpy as np
import amct_onnx
import onnxruntime as ort

NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1"))
KV_CACHE_LENGTH = int(os.environ.get("KV_CACHE_LENGTH", "4096"))
MODEL_PATH = os.environ.get("MODEL_PATH", "output/onnx_changed_310b/DeepSeek-R1-Distill-Qwen-1.5B_4096.onnx")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/amct_onnx_ptq")
HF_MODEL_DIR = os.environ.get("HF_MODEL_DIR", "/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B")
CALIB_FILE = "/usr/local/Ascend/atb-models/examples/convert/model_slim/boolq.jsonl"

print(f"[INFO] ort version: {ort.__version__}, providers: {ort.get_available_providers()}")

config_file = os.path.join(OUTPUT_DIR, "quant_config.json")
modified_model = os.path.join(OUTPUT_DIR, "model_modified.onnx")
record_file = os.path.join(OUTPUT_DIR, "record.txt")
save_path = os.path.join(OUTPUT_DIR, "model_deploy")

# Load BoolQ calibration prompts
print(f"[INFO] Loading BoolQ calibration data: {CALIB_FILE}")
prompts = []
with open(CALIB_FILE, "r") as f:
    for line in f:
        item = json.loads(line.strip())
        text = item.get("inputs_pretokenized", "")
        if text:
            prompts.append(text)
        if len(prompts) >= NUM_SAMPLES:
            break
print(f"[INFO] Loaded {len(prompts)} calibration prompts")

# ============================================================
# Step 1: Create quant config
# ============================================================
print("\n[Step 1] Creating quant config...")
amct_onnx.create_quant_config(
    config_file=config_file,
    model_file=MODEL_PATH,
    skip_layers=["/lm_head/MatMul"],
    batch_num=len(prompts),
    activation_offset=True,
)
print(f"  Config saved: {config_file}")

# ============================================================
# Step 2: Quantize model (insert calibration nodes)
# ============================================================
print("\n[Step 2] Inserting calibration nodes into model...")
amct_onnx.quantize_model(
    config_file=config_file,
    model_file=MODEL_PATH,
    modified_onnx_file=modified_model,
    record_file=record_file,
)
print(f"  Modified model: {modified_model}")

# ============================================================
# Step 3: Run calibration inference
# ============================================================
print(f"\n[Step 3] Running calibration inference ({len(prompts)} samples)...")
print(f"  KV cache length: {KV_CACHE_LENGTH}")

custom_op_lib = os.path.join(
    os.path.dirname(amct_onnx.__file__), "custom_op", "libamct_onnx_ops.so"
)
print(f"  Custom op lib: {custom_op_lib}")

sess_options = ort.SessionOptions()
sess_options.register_custom_ops_library(custom_op_lib)

providers = ["CPUExecutionProvider"]
print(f"  Using providers: {providers} (AMCT calibration ops are CPU-only)")

sess = ort.InferenceSession(modified_model, sess_options, providers=providers)
print("  Session created successfully")

# Load tokenizer for proper calibration
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR, trust_remote_code=True)

kv_dim = 112  # num_hidden_layers * 2 * num_key_value_heads = 28*2*2
per_head_dim = 128

start_time = time.time()
for idx, prompt in enumerate(prompts):
    tokens = tokenizer.encode(prompt)
    prompt_len = min(len(tokens), KV_CACHE_LENGTH - 1)

    input_ids = np.array([[tokens[prompt_len - 1]]], dtype=np.int64)
    attention_mask = np.ones((1, 1 + KV_CACHE_LENGTH), dtype=np.int64)
    attention_mask[:, prompt_len:KV_CACHE_LENGTH] = 0
    position_ids = np.array([[prompt_len - 1]], dtype=np.int64)
    np.random.seed(idx)
    past_key_values = np.zeros((1, KV_CACHE_LENGTH, kv_dim, per_head_dim), dtype=np.float16)

    result = sess.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    })
    elapsed = time.time() - start_time
    print(f"  Sample {idx+1}/{len(prompts)} done ({elapsed:.1f}s), logits shape: {result[0].shape}")

total_time = time.time() - start_time
print(f"  Calibration completed in {total_time:.1f}s")

# Check record
with open(record_file) as f:
    content = f.read()
has_scale_d = "scale_d" in content
print(f"  record has scale_d (activation scale): {has_scale_d}")

if not has_scale_d:
    print("[ERROR] Missing scale_d in record! Activation calibration failed.")
    print("  This means the AMCT custom ops did not collect activation statistics.")
    print("  The record only contains weight scales (scale_w).")
    sys.exit(1)

# ============================================================
# Step 4: Save deploy model
# ============================================================
print("\n[Step 4] Saving deploy model...")
amct_onnx.save_model(modified_model, record_file, save_path)
print(f"  Deploy model: {save_path}_deploy_model.onnx")
print(f"  FakeQuant model: {save_path}_fake_quant_model.onnx")
print("\n[DONE] AMCT quantization complete!")
PYTHON_EOF

PYTHON_EXIT=$?
if [ $PYTHON_EXIT -ne 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "[ERROR] Python 量化流程失败 (exit=$PYTHON_EXIT)" | tee -a "$LOG_FILE"
    echo "日志: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

# ============================================================
# Step 5: ATC 编译 OM (如果 deploy model 存在)
# ============================================================
DEPLOY_ONNX="${OUTPUT_DIR}/model_deploy_deploy_model.onnx"
if [ -f "$DEPLOY_ONNX" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo ">>> 开始 ATC 编译..." | tee -a "$LOG_FILE"
    python3 export/onnx2om.py \
        --hf_model_dir="$HF_MODEL_DIR" \
        --onnx_model_path="$DEPLOY_ONNX" \
        --om_model_path="$OM_OUTPUT" \
        --kv_cache_length="$KV_CACHE_LENGTH" \
        --max_prefill_length=1 \
        --max_batch=1 \
        --cpu_thread=16 \
        --soc_version=auto \
        2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo ">>> OM 编译完成: ${OM_OUTPUT}.om" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "[SKIP] Deploy ONNX 不存在，跳过 ATC 编译" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 完成! 日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
