"""
Profiling 采集脚本：加载 OM 模型，采集推理 profiling 数据。

用法:
    python profiling_collect.py \
        --om_model_path ./output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om \
        --hf_model_dir /mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B \
        --output_dir ./profiling_output/xxx/raw \
        --device_id 0 \
        --kv_cache_length 4096 \
        --max_prefill_length 1 \
        --max_new_tokens 20
"""

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import os
import sys
import time

from config import InferenceConfig
from utils.inference import Inference


def parse_args():
    parser = argparse.ArgumentParser(description="Profiling 数据采集")
    parser.add_argument("--om_model_path", type=str, required=True)
    parser.add_argument("--hf_model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--kv_cache_length", type=int, default=4096)
    parser.add_argument("--max_prefill_length", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = InferenceConfig(
        hf_model_dir=args.hf_model_dir,
        om_model_path=args.om_model_path,
        onnx_model_path="",
        session_type="acl",
        device_id=args.device_id,
        max_batch=1,
        max_input_length=args.kv_cache_length - 1,
        max_output_length=args.kv_cache_length,
        kv_cache_length=args.kv_cache_length,
        max_prefill_length=args.max_prefill_length,
        dtype="float16",
        torch_dtype="float16",
        device_str="npu",
        temperature=0,
        sampling_method="greedy",
        sampling_value=0.95,
        system_prompt="",
    )

    print("[Collect] 加载模型...")
    infer_engine = Inference(config)
    print("[Collect] 模型加载完成")

    print("[Collect] Warmup...")
    for text in infer_engine.stream_predict(prompt="hi", max_new_tokens=3):
        pass
    print("[Collect] Warmup 完成")

    import acl

    ACL_PROF_ACL_API = 0x0001
    ACL_PROF_TASK_TIME = 0x0002
    ACL_PROF_AICORE_METRICS = 0x0004
    ACL_PROF_AICPU_TRACE = 0x0008

    profiling_dir = os.path.abspath(args.output_dir)
    print(f"[Collect] Profiling 输出目录: {profiling_dir}")

    ret = acl.prof.init(profiling_dir)
    if ret != 0:
        print(f"[ERROR] acl.prof.init 失败, ret={ret}")
        sys.exit(1)

    prof_config = acl.prof.create_config(
        [args.device_id], 0, 0,
        ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS | ACL_PROF_AICPU_TRACE,
    )

    ret = acl.prof.start(prof_config)
    if ret != 0:
        print(f"[ERROR] acl.prof.start 失败, ret={ret}")
        acl.prof.destroy_config(prof_config)
        acl.prof.finalize()
        sys.exit(1)

    print("[Collect] === 开始推理 (Profiling 采集中) ===")
    t_start = time.time()
    generated_text = ""
    token_count = 0
    for text in infer_engine.stream_predict(prompt=args.prompt, max_new_tokens=args.max_new_tokens):
        generated_text += text
        token_count += 1
    t_end = time.time()
    duration = t_end - t_start

    acl.prof.stop(prof_config)
    acl.prof.destroy_config(prof_config)

    del infer_engine
    acl.prof.finalize()

    print(f"[Collect] === 推理完成 ===")
    print(f"[Collect] 生成 {token_count} tokens, 耗时 {duration:.3f}s")
    if token_count > 0:
        print(f"[Collect] 平均 {duration / token_count * 1000:.1f} ms/token")
    print(f"[Collect] 输出: {generated_text!r}")
    print(f"[Collect] 原始数据已保存到: {profiling_dir}")


if __name__ == "__main__":
    main()
