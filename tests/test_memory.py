import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""测试 CPU vs NPU 采样的显存占用"""
import sys
import math
import subprocess
import time
from config import InferenceConfig
from utils.inference import Inference

MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
HF_MODEL_DIR = "../models/" + MODEL_NAME
OM_MODEL_PATH = "output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om"
ONNX_MODEL_PATH = "output/onnx/qwen2_1.5b_chat.onnx"


def get_npu_memory():
    """通过 npu-smi 获取当前 NPU 显存占用 (MB)"""
    import re
    result = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True
    )
    for line in result.stdout.split("\n"):
        match = re.search(r'(\d+)\s*/\s*(\d+)\s*$', line.strip().rstrip("|").strip())
        if match:
            return int(match.group(1))
    return -1


def run_test(sampling_device):
    print(f"\n{'='*60}")
    print(f"  测试采样方式: sampling_device={sampling_device}")
    print(f"{'='*60}")

    mem_before = get_npu_memory()
    print(f"[1] 加载模型前显存: {mem_before} MB")

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
        sampling_method="top_p",
        sampling_value=0.95,
        system_prompt="",
        device_str="npu",
        device_id=0,
        sampling_device=sampling_device,
    )

    infer_engine = Inference(config)
    mem_after_load = get_npu_memory()
    print(f"[2] 模型加载后显存: {mem_after_load} MB (增加 {mem_after_load - mem_before} MB)")

    # 跑一次推理
    print("[3] 开始推理: '你好，请用一句话介绍自己'")
    response = ""
    for new_text, ftl, ds, ts in infer_engine.stream_predict(
        "你好，请用一句话介绍自己",
        history=[],
        max_new_tokens=64,
        do_speed_test=True,
    ):
        response += new_text

    mem_after_infer = get_npu_memory()
    print(f"\n[4] 推理完成后显存: {mem_after_infer} MB (增加 {mem_after_infer - mem_before} MB)")
    print(f"    模型回复: {response[:100]}")

    # 再跑一轮确认稳定态
    response2 = ""
    for new_text, ftl, ds, ts in infer_engine.stream_predict(
        "1+1等于几？",
        history=[],
        max_new_tokens=32,
        do_speed_test=True,
    ):
        response2 += new_text

    mem_stable = get_npu_memory()
    print(f"[5] 第二轮推理后显存: {mem_stable} MB (总增加 {mem_stable - mem_before} MB)")

    print(f"\n>>> 结论: sampling_device={sampling_device} 稳定态显存占用 = {mem_stable} MB")
    print(f">>> 净增显存 = {mem_stable - mem_before} MB")

    del infer_engine
    import gc
    gc.collect()
    time.sleep(2)

    return {
        "sampling_device": sampling_device,
        "mem_before": mem_before,
        "mem_after_load": mem_after_load,
        "mem_after_infer": mem_after_infer,
        "mem_stable": mem_stable,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 test_memory.py cpu|npu")
        sys.exit(1)
    device = sys.argv[1]
    assert device in ("cpu", "npu"), "参数必须是 cpu 或 npu"
    result = run_test(device)
    print(f"\n最终结果: {result}")
