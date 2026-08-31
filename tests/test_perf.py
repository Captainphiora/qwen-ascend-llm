import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""性能对比: CPU采样 vs NPU采样 (greedy模式)"""
import sys
import time
import numpy as np
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import InferenceConfig
from utils.inference import Inference

MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR = "../models/" + MODEL_NAME
OM_MODEL_PATH = "output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om"
ONNX_MODEL_PATH = "output/onnx/qwen2_1.5b_chat.onnx"

PROMPTS = ["1+1等于几"]
MAX_NEW_TOKENS = 32


def run_benchmark(sampling_device):
    print(f"\n{'='*60}")
    print(f"  性能测试: sampling_device={sampling_device}, method=greedy")
    print(f"{'='*60}")

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
        sampling_device=sampling_device,
    )

    infer_engine = Inference(config)

    # 直接测试，不warmup

    results = []
    for prompt in PROMPTS:
        print(f"\n[测试] prompt='{prompt}', max_new_tokens={MAX_NEW_TOKENS}")
        first_token_latency = 0
        decode_speed = 0
        total_speed = 0
        token_count = 0
        response = ""

        t_start = time.time()
        for new_text, ftl, ds, ts in infer_engine.stream_predict(
            prompt, history=[], max_new_tokens=MAX_NEW_TOKENS, do_speed_test=True
        ):
            response += new_text
            first_token_latency = ftl
            decode_speed = ds
            total_speed = ts
            token_count += 1
        t_end = time.time()

        print(f"  回复长度: {len(response)} 字符")
        print(f"  首token延迟: {first_token_latency:.4f}s")
        print(f"  decode速度: {decode_speed:.2f} token/s")
        print(f"  总速度(prefill+decode): {total_speed:.2f} token/s")
        print(f"  总耗时: {t_end - t_start:.2f}s")

        results.append({
            "prompt": prompt,
            "first_token_latency": first_token_latency,
            "decode_speed": decode_speed,
            "total_speed": total_speed,
            "total_time": t_end - t_start,
        })

    print(f"\n{'='*60}")
    print(f"  汇总: sampling_device={sampling_device}")
    print(f"{'='*60}")
    avg_ftl = np.mean([r["first_token_latency"] for r in results])
    avg_decode = np.mean([r["decode_speed"] for r in results])
    avg_total = np.mean([r["total_speed"] for r in results])
    print(f"  平均首token延迟: {avg_ftl:.4f}s")
    print(f"  平均decode速度: {avg_decode:.2f} token/s")
    print(f"  平均总速度: {avg_total:.2f} token/s")

    del infer_engine
    import gc
    gc.collect()
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 test_perf.py cpu|npu")
        sys.exit(1)
    device = sys.argv[1]
    assert device in ("cpu", "npu")
    run_benchmark(device)
