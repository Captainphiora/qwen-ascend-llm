"""
详细 Profiling 脚本：分析 Host/Device 执行分布、H2D/D2H 数据搬运、算力利用率

采集内容:
  - AI Core 算子耗时与利用率 (TOPS/TFLOPS)
  - Host 侧耗时 (采样/tokenizer/Python 开销)
  - Device 侧耗时 (模型推理)
  - H2D / D2H 数据搬运量与耗时
  - 显存占用统计

用法:
  python profile_sampling.py --device_id 7
  python profile_sampling.py --device_id 7 --use_msprof

解析 profiling 数据:
  msprof --export=on --output=./profiling_sampling_data/
"""

import argparse
import sys
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import List

DEFAULT_HF_MODEL_DIR = "/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_OM_MODEL_PATH = "output/model_910_cann900/DeepSeek-R1-Distill-Qwen-1.5B_4096_1.om"


@dataclass
class ProfileResult:
    total_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    decode_steps: int = 0
    host_sampling_time_ms: float = 0.0
    host_tokenizer_time_ms: float = 0.0
    host_other_time_ms: float = 0.0
    device_inference_time_ms: float = 0.0
    h2d_time_ms: float = 0.0
    d2h_time_ms: float = 0.0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    npu_sampling_time_ms: float = 0.0
    peak_device_mem_mb: float = 0.0


def profile_inference(infer_engine, prompt, max_new_tokens, sampling_method,
                      sampling_value, temperature) -> ProfileResult:
    """Profile a complete inference run with detailed timing breakdown."""
    import torch
    try:
        import torch_npu
        has_npu = True
    except (ImportError, RuntimeError):
        has_npu = False

    result = ProfileResult()
    session = infer_engine.session

    # Tokenize
    t_tok_start = time.perf_counter()
    messages = [{"role": "user", "content": prompt}]
    text = infer_engine.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = infer_engine.tokenizer(
        [text], return_tensors="np"
    )["input_ids"].astype(np.int64).reshape(1, -1)
    input_ids = input_ids[:, -infer_engine.max_input_length:]
    t_tok_end = time.perf_counter()
    result.host_tokenizer_time_ms = (t_tok_end - t_tok_start) * 1000

    input_length = input_ids.shape[1]
    max_output_len = min(infer_engine.max_output_length - input_length, max_new_tokens)

    ids_list = []
    current_input_ids = input_ids
    total_sampling_time = 0.0
    total_inference_time = 0.0
    total_h2d_time = 0.0
    total_d2h_time = 0.0

    # Estimate data transfer sizes
    vocab_size = infer_engine.tokenizer.vocab_size
    logits_bytes_per_step = vocab_size * 2  # fp16
    input_ids_bytes_prefill = input_length * 8  # int64
    input_ids_bytes_decode = 8  # 1 token int64

    t_start = time.perf_counter()

    for i in range(max_output_len):
        if i == 0:
            session.reset()

        # Model inference (includes H2D of input_ids, D2H of logits internally)
        t_infer_start = time.perf_counter()
        logits = session.run(current_input_ids)
        t_infer_end = time.perf_counter()

        if i == 0:
            result.prefill_time_ms = (t_infer_end - t_infer_start) * 1000

        # Sampling
        t_sample_start = time.perf_counter()
        next_token = infer_engine.sample_logits(
            logits[0][-1:], sampling_method, sampling_value, temperature
        )
        t_sample_end = time.perf_counter()

        next_token = next_token.reshape(1, -1)
        token_id = next_token[0, 0]

        if i > 0:
            total_inference_time += (t_infer_end - t_infer_start)
            total_sampling_time += (t_sample_end - t_sample_start)

        if token_id == infer_engine.tokenizer.eos_token_id:
            break

        ids_list.append(int(token_id))
        current_input_ids = next_token

    t_end = time.perf_counter()

    # Compute results
    result.total_time_ms = (t_end - t_start) * 1000
    result.decode_steps = len(ids_list) - 1 if len(ids_list) > 1 else 0
    result.device_inference_time_ms = total_inference_time * 1000
    result.host_sampling_time_ms = total_sampling_time * 1000

    # Estimate H2D/D2H (model inference includes these internally)
    # H2D: input_ids (prefill: input_length*8 bytes, decode: 1*8 bytes per step)
    #       + attention_mask + position_ids + kv_cache updates
    # D2H: logits (vocab_size * 2 bytes per step)
    result.h2d_bytes = input_ids_bytes_prefill + result.decode_steps * input_ids_bytes_decode
    result.d2h_bytes = (result.decode_steps + 1) * logits_bytes_per_step

    # If NPU sampling is used, logits are also sent H2D for sampling
    if infer_engine.use_npu_sampling and sampling_method != "greedy":
        result.h2d_bytes += result.decode_steps * logits_bytes_per_step
        result.npu_sampling_time_ms = total_sampling_time * 1000

    # Host other time (everything not sampling or inference)
    result.host_other_time_ms = (
        result.total_time_ms
        - result.prefill_time_ms
        - result.device_inference_time_ms
        - result.host_sampling_time_ms
    )

    # Get device memory info
    if has_npu:
        try:
            mem_allocated = torch.npu.memory_allocated() / 1024 / 1024
            mem_reserved = torch.npu.memory_reserved() / 1024 / 1024
            result.peak_device_mem_mb = mem_reserved
        except Exception:
            pass

    return result


def print_profile_report(results: dict, args):
    """Print detailed profiling report."""
    print("\n" + "=" * 95)
    print(f" 详细 Profiling 报告 | Device: npu:{args.device_id} | Model: {os.path.basename(args.om_model_path)}")
    print("=" * 95)

    for label, result in results.items():
        print(f"\n{'─' * 95}")
        print(f" [{label}]")
        print(f"{'─' * 95}")
        print(f"  生成 tokens: {result.decode_steps + 1}, 总耗时: {result.total_time_ms:.1f}ms")
        if result.decode_steps > 0:
            print(f"  Decode 速度: {result.decode_steps / (result.device_inference_time_ms + result.host_sampling_time_ms) * 1000:.1f} tok/s")

        print(f"\n  ┌── 耗时分解 (Decode 阶段, 不含 Prefill) {'─' * 40}┐")
        decode_total = result.device_inference_time_ms + result.host_sampling_time_ms + max(result.host_other_time_ms, 0)
        if decode_total > 0:
            print(f"  │ Device 推理:     {result.device_inference_time_ms:>8.2f} ms  "
                  f"({result.device_inference_time_ms/decode_total*100:>5.1f}%)  "
                  f"avg {result.device_inference_time_ms/max(result.decode_steps,1):.2f} ms/token │")
            print(f"  │ 采样:            {result.host_sampling_time_ms:>8.2f} ms  "
                  f"({result.host_sampling_time_ms/decode_total*100:>5.1f}%)  "
                  f"avg {result.host_sampling_time_ms/max(result.decode_steps,1):.2f} ms/token │")
            print(f"  │ Host 其他:       {max(result.host_other_time_ms, 0):>8.2f} ms  "
                  f"({max(result.host_other_time_ms,0)/decode_total*100:>5.1f}%)                         │")
        print(f"  │ Prefill:          {result.prefill_time_ms:>8.2f} ms                                       │")
        print(f"  │ Tokenizer:        {result.host_tokenizer_time_ms:>8.2f} ms                                       │")
        print(f"  └{'─' * 80}┘")

        print(f"\n  ┌── 数据搬运估算 {'─' * 56}┐")
        print(f"  │ H2D (Host→Device): {result.h2d_bytes/1024:>8.1f} KB  "
              f"(input_ids{' + logits for NPU sampling' if result.npu_sampling_time_ms > 0 else ''}) │")
        print(f"  │ D2H (Device→Host): {result.d2h_bytes/1024:>8.1f} KB  "
              f"(logits, {result.d2h_bytes//(result.decode_steps+1)/1024:.1f}KB/step)               │")
        total_transfer = (result.h2d_bytes + result.d2h_bytes) / 1024 / 1024
        print(f"  │ 总搬运量:          {total_transfer:>8.2f} MB                                         │")
        print(f"  └{'─' * 80}┘")

        # Hardware utilization estimate
        print(f"\n  ┌── 硬件利用率估算 {'─' * 54}┐")
        # 910: 320 TFLOPS FP16, HBM bandwidth ~1.2TB/s
        peak_tflops = 320.0  # FP16 for 910
        peak_bw_gbs = 1200.0  # HBM bandwidth GB/s
        if result.decode_steps > 0:
            avg_infer_ms = result.device_inference_time_ms / result.decode_steps
            # Model params ~1.5B, FP16 = 3GB, each decode step: 2*params FLOPS (matmul)
            model_params = 1.5e9
            flops_per_step = 2 * model_params  # ~3 GFLOPS per decode step
            achieved_tflops = flops_per_step / (avg_infer_ms / 1000) / 1e12
            utilization = achieved_tflops / peak_tflops * 100

            # Memory bandwidth: need to read all weights once per step
            weight_bytes = model_params * 2  # FP16
            achieved_bw = weight_bytes / (avg_infer_ms / 1000) / 1e9
            bw_utilization = achieved_bw / peak_bw_gbs * 100

            print(f"  │ 平台峰值算力:   {peak_tflops:.0f} TFLOPS (FP16)                               │")
            print(f"  │ 实际算力:        {achieved_tflops:.2f} TFLOPS (FP16)                              │")
            print(f"  │ 算力利用率:      {utilization:.1f}%                                            │")
            print(f"  │ HBM 峰值带宽:   {peak_bw_gbs:.0f} GB/s                                         │")
            print(f"  │ 实际带宽:        {achieved_bw:.1f} GB/s                                         │")
            print(f"  │ 带宽利用率:      {bw_utilization:.1f}%                                            │")
            print(f"  │ 瓶颈分析:        {'带宽瓶颈 (Memory-bound)' if bw_utilization > utilization else '计算瓶颈 (Compute-bound)'}          │")
        print(f"  └{'─' * 80}┘")


def main():
    parser = argparse.ArgumentParser(description="详细 Profiling 脚本")
    parser.add_argument("--device_id", type=int, default=7)
    parser.add_argument("--om_model_path", type=str, default=DEFAULT_OM_MODEL_PATH)
    parser.add_argument("--hf_model_dir", type=str, default=DEFAULT_HF_MODEL_DIR)
    parser.add_argument("--kv_cache_length", type=int, default=4096)
    parser.add_argument("--max_prefill_length", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="请详细介绍一下机器学习的基本概念和常用算法")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--sampling_method", type=str, default="all",
                        choices=["all", "greedy", "top_p", "top_k"])
    parser.add_argument("--use_msprof", action="store_true",
                        help="同时采集 ACL profiling 原始数据 (用于 msprof 分析)")
    parser.add_argument("--profiling_dir", type=str, default="./profiling_sampling_data")
    args = parser.parse_args()

    from config import InferenceConfig
    from utils.inference import Inference

    print(f"[Profile] 加载模型 (device_id={args.device_id})...")
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
    print(f"[Profile] NPU sampling: {infer_engine.use_npu_sampling}")

    # Warmup
    print("[Profile] Warmup...")
    for _ in range(2):
        profile_inference(infer_engine, "hi", 5, "greedy", 0.8, 0)

    # Optional: ACL profiling
    acl_profiling = False
    if args.use_msprof:
        try:
            import acl
            profiling_dir = os.path.abspath(args.profiling_dir)
            os.makedirs(profiling_dir, exist_ok=True)

            ACL_PROF_ACL_API = 0x0001
            ACL_PROF_TASK_TIME = 0x0002
            ACL_PROF_AICORE_METRICS = 0x0004
            ACL_PROF_AICPU_TRACE = 0x0008

            ret = acl.prof.init(profiling_dir)
            if ret == 0:
                prof_config = acl.prof.create_config(
                    [args.device_id], 0, 0,
                    ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS | ACL_PROF_AICPU_TRACE
                )
                ret = acl.prof.start(prof_config)
                if ret == 0:
                    acl_profiling = True
                    print(f"[Profile] ACL profiling 已启动, 输出: {profiling_dir}")
        except Exception as e:
            print(f"[Profile] ACL profiling 启动失败: {e}")

    # Run profiling
    test_configs = []
    if args.sampling_method == "all":
        test_configs = [
            ("Greedy (CPU)", "greedy", 0.8, 0.0, True),
            ("Top-p=0.8 (NPU ATB)", "top_p", 0.8, 0.7, True),
            ("Top-p=0.8 (CPU numpy)", "top_p", 0.8, 0.7, False),
            ("Top-k=50 (NPU ATB)", "top_k", 50, 0.7, True),
            ("Top-k=50 (CPU numpy)", "top_k", 50, 0.7, False),
        ]
    else:
        method = args.sampling_method
        val = 0.8 if method == "top_p" else 50
        temp = 0.0 if method == "greedy" else 0.7
        test_configs = [
            (f"{method} (NPU)", method, val, temp, True),
            (f"{method} (CPU)", method, val, temp, False),
        ]

    results = {}
    for label, method, value, temp, use_npu in test_configs:
        if use_npu:
            infer_engine.use_npu_sampling = True
        else:
            infer_engine.use_npu_sampling = False

        result = profile_inference(infer_engine, args.prompt, args.max_new_tokens, method, value, temp)
        results[label] = result
        print(f"  {label:<30} done ({result.decode_steps + 1} tokens)")

    # Stop ACL profiling
    if acl_profiling:
        acl.prof.stop(prof_config)
        acl.prof.destroy_config(prof_config)
        acl.prof.finalize()
        print(f"[Profile] ACL profiling 数据已保存到: {profiling_dir}")

    # Print report
    print_profile_report(results, args)

    # Summary recommendation
    print("\n" + "=" * 95)
    print(" 总结与建议")
    print("=" * 95)
    greedy_result = results.get("Greedy (CPU)")
    npu_topp = results.get("Top-p=0.8 (NPU ATB)")
    cpu_topp = results.get("Top-p=0.8 (CPU numpy)")
    if greedy_result and npu_topp and cpu_topp:
        print(f"  - Greedy decode: {greedy_result.decode_steps / (greedy_result.device_inference_time_ms + greedy_result.host_sampling_time_ms) * 1000:.1f} tok/s")
        print(f"  - Top-p NPU:     {npu_topp.decode_steps / (npu_topp.device_inference_time_ms + npu_topp.host_sampling_time_ms) * 1000:.1f} tok/s")
        print(f"  - Top-p CPU:     {cpu_topp.decode_steps / (cpu_topp.device_inference_time_ms + cpu_topp.host_sampling_time_ms) * 1000:.1f} tok/s")
        npu_gain = (1 - cpu_topp.host_sampling_time_ms / max(npu_topp.host_sampling_time_ms, 0.001)) * 100
        print(f"  - NPU 采样 vs CPU: 采样耗时减少 {cpu_topp.host_sampling_time_ms - npu_topp.host_sampling_time_ms:.1f}ms ({abs(npu_gain):.0f}%)")
    print()


if __name__ == "__main__":
    main()
