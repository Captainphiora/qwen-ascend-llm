"""
采样策略性能对比基准测试
对比 NPU ATB 采样 vs CPU numpy 采样在 greedy/top_p/top_k 下的完整推理性能。

指标：TTFT, TPOT, Decode Speed, 采样耗时占比

用法:
  python benchmark_sampling.py --device_id 7
  python benchmark_sampling.py --device_id 7 --max_new_tokens 100 --rounds 3
"""

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import sys
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List

DEFAULT_HF_MODEL_DIR = "/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_OM_MODEL_PATH = "output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om"


@dataclass
class SamplingBenchResult:
    label: str = ""
    sampling_method: str = ""
    sampling_backend: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    decode_speed: float = 0.0
    total_time_ms: float = 0.0
    sampling_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    sampling_ratio: float = 0.0


def run_benchmark(infer_engine, prompt, max_new_tokens, sampling_method, sampling_value, temperature) -> SamplingBenchResult:
    """Run single benchmark measuring inference and sampling time separately."""
    messages = [{"role": "user", "content": prompt}]
    text = infer_engine.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = infer_engine.tokenizer(
        [text], return_tensors="np"
    )["input_ids"].astype(np.int64).reshape(1, -1)
    input_ids = input_ids[:, -infer_engine.max_input_length:]

    input_length = input_ids.shape[1]
    max_output_len = min(infer_engine.max_output_length - input_length, max_new_tokens)
    session = infer_engine.session

    ids_list = []
    current_input_ids = input_ids
    total_sampling_time = 0.0
    total_inference_time = 0.0

    t_start = time.perf_counter()
    t_first_token = None

    for i in range(max_output_len):
        if i == 0:
            session.reset()

        t_infer_start = time.perf_counter()
        logits = session.run(current_input_ids)
        t_infer_end = time.perf_counter()

        if i == 0:
            t_first_token = time.perf_counter()

        t_sample_start = time.perf_counter()
        last_logits = infer_engine._get_last_logits(logits)
        next_token = infer_engine.sample_logits(
            last_logits, sampling_method, sampling_value, temperature
        )
        t_sample_end = time.perf_counter()

        next_token = next_token.reshape(1, -1)
        token_id = next_token[0, 0]

        if i > 0:
            total_sampling_time += (t_sample_end - t_sample_start)
            total_inference_time += (t_infer_end - t_infer_start)

        if token_id == infer_engine.tokenizer.eos_token_id:
            break

        ids_list.append(int(token_id))
        current_input_ids = next_token

    t_end = time.perf_counter()

    result = SamplingBenchResult()
    result.sampling_method = sampling_method
    result.sampling_backend = "NPU ATB" if infer_engine.use_npu_sampling and sampling_method != "greedy" else "CPU numpy"
    result.prompt_tokens = input_length
    result.generated_tokens = len(ids_list)
    result.total_time_ms = (t_end - t_start) * 1000

    if t_first_token:
        result.ttft_ms = (t_first_token - t_start) * 1000

    decode_tokens = result.generated_tokens - 1
    if decode_tokens > 0 and t_first_token:
        decode_time = (t_end - t_first_token) * 1000
        result.tpot_ms = decode_time / decode_tokens
        result.decode_speed = decode_tokens / (decode_time / 1000)
        result.sampling_time_ms = total_sampling_time * 1000
        result.inference_time_ms = total_inference_time * 1000
        if (total_sampling_time + total_inference_time) > 0:
            result.sampling_ratio = total_sampling_time / (total_sampling_time + total_inference_time) * 100

    return result


def print_comparison(all_results: dict, args):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print(f" 采样策略性能对比 | Device: npu:{args.device_id} | Model: {os.path.basename(args.om_model_path)}")
    print("=" * 90)
    print(f" Prompt: {args.prompt!r}")
    print(f" Max tokens: {args.max_new_tokens}, Rounds: {args.rounds}")
    print()

    header = f"{'配置':<30} {'TTFT(ms)':<10} {'TPOT(ms)':<10} {'Decode(tok/s)':<14} {'采样耗时(ms)':<13} {'采样占比':<10}"
    print(header)
    print("-" * 90)

    for label, results in all_results.items():
        avg_ttft = np.mean([r.ttft_ms for r in results])
        avg_tpot = np.mean([r.tpot_ms for r in results if r.tpot_ms > 0])
        avg_decode = np.mean([r.decode_speed for r in results if r.decode_speed > 0])
        avg_sampling = np.mean([r.sampling_time_ms for r in results])
        avg_ratio = np.mean([r.sampling_ratio for r in results])
        print(f"{label:<30} {avg_ttft:<10.1f} {avg_tpot:<10.2f} {avg_decode:<14.1f} {avg_sampling:<13.2f} {avg_ratio:<10.1f}%")

    print("-" * 90)
    print()


def main():
    parser = argparse.ArgumentParser(description="采样策略性能对比基准测试")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--om_model_path", type=str, default=DEFAULT_OM_MODEL_PATH)
    parser.add_argument("--hf_model_dir", type=str, default=DEFAULT_HF_MODEL_DIR)
    parser.add_argument("--kv_cache_length", type=int, default=4096)
    parser.add_argument("--max_prefill_length", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="请详细介绍一下机器学习的基本概念和常用算法")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--npu-only", action="store_true", default=False,
                        help="只运行 NPU 采样测试，跳过 CPU 采样")
    args = parser.parse_args()

    from config import InferenceConfig
    from utils.inference import Inference, HAS_TORCH_NPU

    print(f"[Bench] 加载模型 (device_id={args.device_id})...")
    config = InferenceConfig(
        hf_model_dir=args.hf_model_dir,
        om_model_path=args.om_model_path,
        onnx_model_path="",
        session_type="acl",
        device_id=args.device_id,
        max_batch=1,
        max_input_length=4095,
        max_output_length=4096,
        kv_cache_length=args.kv_cache_length,
        max_prefill_length=args.max_prefill_length,
        dtype="float16",
        torch_dtype="float16",
        device_str="npu",
        temperature=0,
        sampling_method="greedy",
        sampling_value=0.8,
        system_prompt="",
    )
    infer_engine = Inference(config)
    npu_sampling_available = HAS_TORCH_NPU and config.session_type == "acl"
    print(f"[Bench] NPU sampling available: {npu_sampling_available}")
    print(f"[Bench] NPU sampling enabled (env USE_NPU_SAMPLING): {infer_engine.use_npu_sampling}")

    # Warmup
    print(f"[Bench] Warmup ({args.warmup} rounds)...")
    for _ in range(args.warmup):
        run_benchmark(infer_engine, "hi", 5, "greedy", 0.8, 0)

    # Test configurations
    # NPU ATB 测试需要设置 USE_NPU_SAMPLING=1 (会产生退出时的 harmless warning)
    test_configs = []
    if not args.npu_only:
        test_configs = [
            ("Greedy (CPU argmax)", "greedy", 0.8, 0.0, False),
            ("Top-p=0.8 (CPU numpy)", "top_p", 0.8, 0.7, False),
            ("Top-p=0.95 (CPU numpy)", "top_p", 0.95, 0.7, False),
            ("Top-k=50 (CPU numpy)", "top_k", 50, 0.7, False),
        ]
    if npu_sampling_available and infer_engine.use_npu_sampling:
        test_configs.extend([
            ("Top-p=0.8 (NPU ATB)", "top_p", 0.8, 0.7, True),
            ("Top-p=0.95 (NPU ATB)", "top_p", 0.95, 0.7, True),
            ("Top-k=50 (NPU ATB)", "top_k", 50, 0.7, True),
        ])

    all_results = {}
    for label, method, value, temp, use_npu in test_configs:
        infer_engine.sampling_method = method
        infer_engine.temperature = temp
        if use_npu:
            infer_engine.use_npu_sampling = True
            infer_engine.session.model._skip_logits_d2h = True
        else:
            infer_engine.use_npu_sampling = False
            infer_engine.session.model._skip_logits_d2h = False

        results = []
        for r in range(args.rounds):
            result = run_benchmark(infer_engine, args.prompt, args.max_new_tokens, method, value, temp)
            results.append(result)
        all_results[label] = results
        avg_decode = np.mean([r.decode_speed for r in results if r.decode_speed > 0])
        print(f"  {label:<30} -> {avg_decode:.1f} tok/s")

    # Restore npu sampling
    if npu_sampling_available:
        infer_engine.use_npu_sampling = True
        infer_engine.session.model._skip_logits_d2h = True

    print_comparison(all_results, args)


if __name__ == "__main__":
    main()
