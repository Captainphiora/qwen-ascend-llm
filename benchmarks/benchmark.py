"""
性能基准测试脚本：测量 TTFT、TPOT、吞吐量等关键指标
支持多轮测试取平均值，方便对比不同优化方案。

指标说明:
  TTFT  (Time To First Token)  : 首字延迟，从输入到第一个token生成的时间
  TPOT  (Time Per Output Token): 每个输出token的平均生成时间（不含首字）
  Throughput                   : 总吞吐量 (tokens/sec)
  Prefill Speed                : Prefill 阶段的处理速度 (tokens/sec)

用法:
  python benchmark.py
  python benchmark.py --prompt "你好" --max_new_tokens 50 --rounds 3
  python benchmark.py --label baseline
  python benchmark.py --label optimized_rope --om_model_path ./output/model_opt/xxx.om
"""

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import sys
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import List

# ============================================================
# 默认配置
# ============================================================
DEFAULT_HF_MODEL_DIR = "/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_OM_MODEL_PATH = "output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om"


DEVICE_ID=0
@dataclass
class BenchmarkResult:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    total_time_ms: float = 0.0
    prefill_speed: float = 0.0
    decode_speed: float = 0.0
    throughput: float = 0.0


def run_single_benchmark(infer_engine, session, prompt, max_new_tokens) -> BenchmarkResult:
    """执行单次推理并测量性能指标"""
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

    ids_list = []
    current_input_ids = input_ids

    t_start = time.perf_counter()
    t_first_token = None

    for i in range(max_output_len):
        if i == 0:
            session.reset()

        logits = session.run(current_input_ids)

        if i == 0:
            t_first_token = time.perf_counter()

        next_token = infer_engine.sample_logits(logits[0][-1:], "greedy", 0.95, 0)
        next_token = next_token.reshape(1, -1)
        token_id = next_token[0, 0]

        if token_id == infer_engine.tokenizer.eos_token_id:
            break

        ids_list.append(int(token_id))
        current_input_ids = next_token

    t_end = time.perf_counter()

    # 计算指标
    result = BenchmarkResult()
    result.prompt_tokens = input_length
    result.generated_tokens = len(ids_list)
    result.total_time_ms = (t_end - t_start) * 1000

    if t_first_token:
        result.ttft_ms = (t_first_token - t_start) * 1000
        result.prefill_speed = input_length / (result.ttft_ms / 1000) if result.ttft_ms > 0 else 0

    if result.generated_tokens > 1 and t_first_token:
        decode_time = (t_end - t_first_token) * 1000
        result.tpot_ms = decode_time / (result.generated_tokens - 1)
        result.decode_speed = (result.generated_tokens - 1) / (decode_time / 1000)

    if result.total_time_ms > 0:
        result.throughput = (result.prompt_tokens + result.generated_tokens) / (result.total_time_ms / 1000)

    return result


def print_results(results: List[BenchmarkResult], label: str = ""):
    """打印性能结果"""
    if not results:
        return

    print()
    print("=" * 70)
    if label:
        print(f" Benchmark Results: {label}")
    else:
        print(" Benchmark Results")
    print("=" * 70)
    print()

    if len(results) > 1:
        print(f"{'轮次':<6} {'TTFT(ms)':<12} {'TPOT(ms)':<12} {'Decode(tok/s)':<14} {'生成tokens':<10}")
        print("-" * 60)
        for i, r in enumerate(results):
            print(f"{i+1:<6} {r.ttft_ms:<12.2f} {r.tpot_ms:<12.2f} {r.decode_speed:<14.1f} {r.generated_tokens:<10}")
        print("-" * 60)

    avg_ttft = np.mean([r.ttft_ms for r in results])
    avg_tpot = np.mean([r.tpot_ms for r in results if r.tpot_ms > 0]) if any(r.tpot_ms > 0 for r in results) else 0
    avg_decode_speed = np.mean([r.decode_speed for r in results if r.decode_speed > 0]) if any(r.decode_speed > 0 for r in results) else 0
    avg_prefill_speed = np.mean([r.prefill_speed for r in results if r.prefill_speed > 0]) if any(r.prefill_speed > 0 for r in results) else 0
    avg_throughput = np.mean([r.throughput for r in results])
    prompt_tokens = results[0].prompt_tokens
    avg_gen_tokens = np.mean([r.generated_tokens for r in results])

    print()
    print(f"  输入长度:          {prompt_tokens} tokens")
    print(f"  平均生成长度:      {avg_gen_tokens:.0f} tokens")
    print(f"  测试轮次:          {len(results)}")
    print()
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │ TTFT (首字延迟):   {avg_ttft:>10.2f} ms        │")
    print(f"  │ TPOT (每token):    {avg_tpot:>10.2f} ms        │")
    print(f"  │ Prefill 速度:      {avg_prefill_speed:>10.1f} tokens/s  │")
    print(f"  │ Decode 速度:       {avg_decode_speed:>10.1f} tokens/s  │")
    print(f"  │ 总吞吐量:          {avg_throughput:>10.1f} tokens/s  │")
    print(f"  └─────────────────────────────────────────┘")
    print()

    if len(results) > 1:
        std_ttft = np.std([r.ttft_ms for r in results])
        std_tpot = np.std([r.tpot_ms for r in results if r.tpot_ms > 0]) if any(r.tpot_ms > 0 for r in results) else 0
        print(f"  TTFT 标准差:       ±{std_ttft:.2f} ms")
        print(f"  TPOT 标准差:       ±{std_tpot:.2f} ms")
        print()


def main():
    parser = argparse.ArgumentParser(description="LLM 推理性能基准测试")
    parser.add_argument("--prompt", type=str, default="请详细介绍一下机器学习的基本概念和常用算法",
                        help="测试 prompt")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="最大生成 token 数")
    parser.add_argument("--rounds", type=int, default=3,
                        help="测试轮数")
    parser.add_argument("--warmup", type=int, default=0,
                        help="预热轮数")
    parser.add_argument("--om_model_path", type=str, default=DEFAULT_OM_MODEL_PATH,
                        help="OM 模型路径")
    parser.add_argument("--hf_model_dir", type=str, default=DEFAULT_HF_MODEL_DIR,
                        help="HuggingFace 模型目录")
    parser.add_argument("--kv_cache_length", type=int, default=4096)
    parser.add_argument("--max_prefill_length", type=int, default=1)
    parser.add_argument("--label", type=str, default="",
                        help="本次测试标签 (如 'baseline' / 'optimized_rope')")
    parser.add_argument("--device_id", type=int, default=0,
                        )
    parser.add_argument("--kv_cache_layout", type=str, default="BSHD", choices=["BSHD", "BHSD"])
    args = parser.parse_args()

    from config import InferenceConfig
    from utils.inference import Inference

    print(f"[Benchmark] 配置:")
    print(f"  OM模型: {args.om_model_path}")
    print(f"  max_prefill_length: {args.max_prefill_length}")
    print(f"  kv_cache_length: {args.kv_cache_length}")
    print(f"[Benchmark] 加载模型...")

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
        sampling_value=0.95,
        system_prompt="",
        kv_cache_layout=args.kv_cache_layout,
    )
    infer_engine = Inference(config)
    session = infer_engine.session
    print("[Benchmark] 模型加载完成")

    # Warmup
    print(f"[Benchmark] Warmup ({args.warmup} 轮)...")
    for _ in range(args.warmup):
        run_single_benchmark(infer_engine, session, "hello", 5)
    print("[Benchmark] Warmup 完成")

    # 正式测试
    print(f"[Benchmark] 开始测试 ({args.rounds} 轮)...")
    print(f"  Prompt: {args.prompt!r}")
    print(f"  Max tokens: {args.max_new_tokens}")

    results = []
    for i in range(args.rounds):
        result = run_single_benchmark(infer_engine, session, args.prompt, args.max_new_tokens)
        results.append(result)
        print(f"  轮 {i+1}: TTFT={result.ttft_ms:.1f}ms, TPOT={result.tpot_ms:.1f}ms, "
              f"generated={result.generated_tokens} tokens")

    print_results(results, label=args.label or os.path.basename(args.om_model_path))


if __name__ == "__main__":
    main()
