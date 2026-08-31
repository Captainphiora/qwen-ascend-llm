import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""测试复用 context 模式 (USE_NPU_SAMPLING=1) 的性能"""
import os
os.environ["USE_NPU_SAMPLING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import numpy as np
from config import InferenceConfig
from utils.inference import Inference

MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR = "../models/" + MODEL_NAME
OM_MODEL_PATH = "output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om"
ONNX_MODEL_PATH = "output/onnx/qwen2_1.5b_chat.onnx"

import subprocess, re
def get_npu_memory():
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        match = re.search(r'(\d+)\s*/\s*(\d+)\s*$', line.strip().rstrip("|").strip())
        if match:
            return int(match.group(1))
    return -1

def run_test():
    print(f"\n{'='*60}")
    print(f"  测试: USE_NPU_SAMPLING=1 (复用context), greedy")
    print(f"{'='*60}")

    mem_before = get_npu_memory()
    print(f"[0] 初始显存: {mem_before} MB")
    print("[0.1] 开始创建 config...", flush=True)

    config = InferenceConfig(
        hf_model_dir=HF_MODEL_DIR,
        om_model_path=OM_MODEL_PATH,
        onnx_model_path=ONNX_MODEL_PATH,
        cpu_thread=1,
        session_type="acl",
        max_batch=1,
        max_output_length=4096,
        max_input_length=1024,
        kv_cache_length=4096,
        max_prefill_length=1,
        dtype="float16",
        torch_dtype="float16",
        temperature=0.6,
        sampling_method="greedy",
        sampling_value=0.95,
        system_prompt="",
        device_str="npu",
        device_id=0,
        sampling_device="npu",
    )

    infer_engine = Inference(config)
    print("[1.1] Inference 初始化完成", flush=True)
    mem_loaded = get_npu_memory()
    print(f"[1] 模型加载后显存: {mem_loaded} MB (+{mem_loaded - mem_before} MB)")

    # 推理
    print("[2] 推理: '1+1等于几'")
    response = ""
    t_start = time.time()
    for new_text, ftl, ds, ts in infer_engine.stream_predict(
        "1+1等于几", history=[], max_new_tokens=16, do_speed_test=True
    ):
        response += new_text
        first_token_latency = ftl
        decode_speed = ds
        total_speed = ts
    t_end = time.time()

    mem_after = get_npu_memory()
    print(f"\n[3] 结果:")
    print(f"    首token延迟: {first_token_latency:.4f}s")
    print(f"    decode速度: {decode_speed:.2f} token/s")
    print(f"    总速度: {total_speed:.2f} token/s")
    print(f"    总耗时: {t_end - t_start:.2f}s")
    print(f"    显存: {mem_after} MB (+{mem_after - mem_before} MB)")
    print(f"    回复: {response[:80]}")

if __name__ == "__main__":
    run_test()
