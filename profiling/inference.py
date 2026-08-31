#!/usr/bin/env python3
"""
Profiling 用推理脚本 (非交互, 单次推理)

直接加载模型并执行一次完整推理, 用于被 msprof 包裹采集数据。
所有参数通过命令行传入 (由 run_inference.sh 构造)。
"""

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import sys
import os
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Profiling 推理")
    parser.add_argument("--hf_model_dir", type=str, required=True)
    parser.add_argument("--om_model_path", type=str, required=True)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--kv_cache_length", type=int, default=4096)
    parser.add_argument("--max_prefill_length", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="请介绍一下机器学习")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sampling_method", type=str, default="greedy")
    parser.add_argument("--sampling_value", type=float, default=0.8)
    args = parser.parse_args()

    from config import InferenceConfig
    from utils.inference import Inference

    config = InferenceConfig(
        hf_model_dir=args.hf_model_dir,
        om_model_path=args.om_model_path,
        onnx_model_path="",
        session_type="acl",
        device_id=args.device_id,
        max_batch=1,
        max_input_length=4095,
        max_output_length=args.kv_cache_length,
        kv_cache_length=args.kv_cache_length,
        max_prefill_length=args.max_prefill_length,
        dtype="float16",
        torch_dtype="float16",
        device_str="npu",
        temperature=args.temperature,
        sampling_method=args.sampling_method,
        sampling_value=args.sampling_value,
        system_prompt="",
    )

    print(f"[Prof] 加载模型 (device_id={args.device_id})...")
    infer_engine = Inference(config)

    messages = [{"role": "user", "content": args.prompt}]
    text = infer_engine.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = infer_engine.tokenizer(
        [text], return_tensors="np"
    )["input_ids"].astype(np.int64).reshape(1, -1)
    input_ids = input_ids[:, -infer_engine.max_input_length:]

    input_length = input_ids.shape[1]
    max_output_len = min(
        infer_engine.max_output_length - input_length, args.max_new_tokens
    )

    print(f"[Prof] 开始推理: input_tokens={input_length}, max_new_tokens={max_output_len}")
    session = infer_engine.session
    session.reset()

    ids_list = []
    current_input_ids = input_ids
    t_start = time.perf_counter()

    for i in range(max_output_len):
        logits = session.run(current_input_ids)
        last_logits = infer_engine._get_last_logits(logits)
        next_token = infer_engine.sample_logits(
            last_logits, args.sampling_method, args.sampling_value, args.temperature
        )
        next_token = next_token.reshape(1, -1)
        token_id = next_token[0, 0]

        if token_id == infer_engine.tokenizer.eos_token_id:
            break

        ids_list.append(int(token_id))
        current_input_ids = next_token

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    output_text = infer_engine.tokenizer.decode(ids_list, skip_special_tokens=True)
    print(f"[Prof] 推理完成: generated_tokens={len(ids_list)}, "
          f"time={elapsed:.3f}s, speed={len(ids_list)/elapsed:.1f} tok/s")
    print(f"[Prof] 输出: {output_text[:200]}...")


if __name__ == "__main__":
    main()
