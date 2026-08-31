"""
Profiling 采集脚本：使用 ACL Python API 在代码内部采集性能数据
不依赖 msprof 外部包裹，直接调用 acl.prof.* 接口。

用法:
    python profile_inference.py

采集完成后，profiling 原始数据保存在 ./profiling_data/ 目录下。
可以用 msprof 解析:
    bash run_profiling.sh --parse
"""

import sys
import os
import time
import numpy as np

# ============================================================
# CONFIG
# ============================================================
HF_MODEL_DIR = "/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
# OM_MODEL_PATH = "./output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om"
OM_MODEL_PATH = "./output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_sim.om"
SESSION_TYPE = "acl"
DEVICE_ID = 5
KV_CACHE_LENGTH = 4096
MAX_PREFILL_LENGTH = 1
MAX_INPUT_LENGTH = 4095
MAX_OUTPUT_LENGTH = 4096
MAX_BATCH = 1
DTYPE = "float16"
TORCH_DTYPE = "float16"
DEVICE_STR = "npu"

TEST_PROMPT = "你好，请介绍一下你自己"
MAX_NEW_TOKENS = 20
TEMPERATURE = 0
SYSTEM_PROMPT = ""

PROFILING_OUTPUT_DIR = "./profiling_data"
# ============================================================

from config import InferenceConfig
from utils.inference import Inference


def main():
    # 准备 profiling 输出目录
    profiling_dir = os.path.abspath(PROFILING_OUTPUT_DIR)
    os.makedirs(profiling_dir, exist_ok=True)

    print("[Profile] 初始化模型...")
    config = InferenceConfig(
        hf_model_dir=HF_MODEL_DIR,
        om_model_path=OM_MODEL_PATH,
        onnx_model_path="",
        session_type=SESSION_TYPE,
        device_id=DEVICE_ID,
        max_batch=MAX_BATCH,
        max_input_length=MAX_INPUT_LENGTH,
        max_output_length=MAX_OUTPUT_LENGTH,
        kv_cache_length=KV_CACHE_LENGTH,
        max_prefill_length=MAX_PREFILL_LENGTH,
        dtype=DTYPE,
        torch_dtype=TORCH_DTYPE,
        device_str=DEVICE_STR,
        temperature=TEMPERATURE,
        sampling_method="greedy",
        sampling_value=0.95,
        system_prompt=SYSTEM_PROMPT,
    )
    infer_engine = Inference(config)
    print("[Profile] 模型加载完成")

    # Warmup
    print("[Profile] Warmup...")
    for text in infer_engine.stream_predict(prompt="hi", max_new_tokens=3):
        pass
    print("[Profile] Warmup 完成")

    # ============================================================
    # 开启 Profiling（使用 ACL Python API）
    # ============================================================
    import acl

    # Profiling 标志位
    ACL_PROF_ACL_API = 0x0001
    ACL_PROF_TASK_TIME = 0x0002
    ACL_PROF_AICORE_METRICS = 0x0004
    ACL_PROF_AICPU_TRACE = 0x0008

    print(f"[Profile] 初始化 Profiling, 数据输出到: {profiling_dir}")
    ret = acl.prof.init(profiling_dir)
    if ret != 0:
        print(f"[ERROR] acl.prof.init 失败, ret={ret}")
        print("[ERROR] 可能原因: 目录权限不足或 CANN 版本不支持")
        return

    # 创建 profiling 配置
    device_list = [DEVICE_ID]
    prof_config = acl.prof.create_config(
        device_list,
        0,    # 采样间隔（0=默认）
        0,    # 输出模式（0=默认）
        ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS | ACL_PROF_AICPU_TRACE
    )

    # 开始采集
    print("[Profile] === 开始 Profiling 采集 ===")
    ret = acl.prof.start(prof_config)
    if ret != 0:
        print(f"[ERROR] acl.prof.start 失败, ret={ret}")
        acl.prof.destroy_config(prof_config)
        acl.prof.finalize()
        return

    # ============================================================
    # 正式推理 (profiling 采集区域)
    # ============================================================
    t_start = time.time()

    generated_text = ""
    token_count = 0
    for text in infer_engine.stream_predict(
        prompt=TEST_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
    ):
        generated_text += text
        token_count += 1

    t_end = time.time()
    duration = t_end - t_start

    # ============================================================
    # 停止 Profiling
    # ============================================================
    ret = acl.prof.stop(prof_config)
    print(f"[Profile] acl.prof.stop ret={ret}")

    ret = acl.prof.destroy_config(prof_config)
    print(f"[Profile] acl.prof.destroy_config ret={ret}")

    ret = acl.prof.finalize()
    print(f"[Profile] acl.prof.finalize ret={ret}")

    # ============================================================
    # 输出结果
    # ============================================================
    print(f"\n[Profile] === 推理完成 ===")
    print(f"[Profile] 输入: {TEST_PROMPT!r}")
    print(f"[Profile] 输出: {generated_text!r}")
    print(f"[Profile] 生成 {token_count} tokens, 耗时 {duration:.3f}s")
    if token_count > 0:
        print(f"[Profile] 平均 {duration/token_count*1000:.1f} ms/token")

    print(f"\n[Profile] Profiling 数据已保存到: {profiling_dir}")
    print(f"[Profile] 目录内容:")
    for item in os.listdir(profiling_dir):
        item_path = os.path.join(profiling_dir, item)
        if os.path.isdir(item_path):
            print(f"  [DIR] {item}")
        else:
            print(f"  [FILE] {item}")

    print(f"\n[Profile] 下一步 - 解析 profiling 数据:")
    print(f"  bash run_profiling.sh --parse")


if __name__ == "__main__":
    main()
