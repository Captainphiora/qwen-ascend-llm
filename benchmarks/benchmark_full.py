"""
综合性能基准测试：一次加载模型，运行所有测试项
测试项：Decode 吞吐、Prefill 延迟（不同输入长度 + 不同 prefill 分块）、采样策略对比
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

HF_MODEL_DIR = "/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B"
OM_MODEL_PATH = "output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om"
DEVICE_ID = 0


@dataclass
class Result:
    label: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    decode_speed: float = 0.0
    prefill_speed: float = 0.0
    throughput: float = 0.0
    total_time_ms: float = 0.0


def run_once(infer_engine, session, prompt, max_new_tokens) -> Result:
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

    r = Result()
    r.prompt_tokens = input_length
    r.generated_tokens = len(ids_list)
    r.total_time_ms = (t_end - t_start) * 1000
    if t_first_token:
        r.ttft_ms = (t_first_token - t_start) * 1000
        r.prefill_speed = input_length / (r.ttft_ms / 1000) if r.ttft_ms > 0 else 0
    if r.generated_tokens > 1 and t_first_token:
        decode_time = (t_end - t_first_token) * 1000
        r.tpot_ms = decode_time / (r.generated_tokens - 1)
        r.decode_speed = (r.generated_tokens - 1) / (decode_time / 1000)
    if r.total_time_ms > 0:
        r.throughput = (r.prompt_tokens + r.generated_tokens) / (r.total_time_ms / 1000)
    return r


def run_test(infer_engine, session, label, prompt, max_new_tokens):
    r = run_once(infer_engine, session, prompt, max_new_tokens)
    r.label = label
    print(f"  {label:<38} prompt={r.prompt_tokens:<5} gen={r.generated_tokens:<6} TTFT={r.ttft_ms:<10.1f} TPOT={r.tpot_ms:<8.1f} Decode={r.decode_speed:<6.1f} tok/s  Prefill={r.prefill_speed:<6.1f} tok/s")
    return r


def create_engine(max_prefill_length=1):
    from config import InferenceConfig
    from utils.inference import Inference
    config = InferenceConfig(
        hf_model_dir=HF_MODEL_DIR,
        om_model_path=OM_MODEL_PATH,
        onnx_model_path="",
        session_type="acl",
        device_id=DEVICE_ID,
        max_batch=1,
        max_input_length=4095,
        max_output_length=4096,
        kv_cache_length=4096,
        max_prefill_length=max_prefill_length,
        dtype="float16",
        torch_dtype="float16",
        device_str="npu",
        temperature=0,
        sampling_method="greedy",
        sampling_value=0.95,
        system_prompt="",
    )
    engine = Inference(config)
    return engine


def check_memory():
    with open('/proc/meminfo') as f:
        lines = f.readlines()
    info = {}
    for line in lines:
        parts = line.split()
        info[parts[0].rstrip(':')] = int(parts[1])
    avail_mb = info.get('MemAvailable', 0) // 1024
    swap_used_kb = info.get('SwapTotal', 0) - info.get('SwapFree', 0)
    return avail_mb, swap_used_kb


def main():
    os.environ['ACL_LOAD_FROM_FILE'] = '1'
    
    print("=" * 100)
    print(" 综合性能基准测试 | 310B1 | v4_noexpand_310b.om")
    print("=" * 100)

    avail, swap = check_memory()
    print(f"\n[环境] MemAvailable={avail}MB, SwapUsed={swap}KB")
    assert swap == 0, "SWAP 已被使用，测试中止"

    short_prompt = "你好"
    medium_prompt = "请详细介绍一下机器学习的基本概念和常用算法，包括监督学习、无监督学习、强化学习的区别和应用场景"
    long_prompt = "机器学习是人工智能的重要分支，它通过数据驱动的方式让计算机系统自动学习和改进。" * 25

    all_results = {}

    # ============ Part 1: Decode 吞吐 (prefill=1) ============
    print("\n" + "=" * 100)
    print(" [Part 1/4] Decode 吞吐基准 (max_prefill_length=1, greedy)")
    print("=" * 100)
    
    engine = create_engine(max_prefill_length=1)
    session = engine.session
    print("[模型加载完成]")

    for tokens in [50, 100, 256]:
        label = f"decode_{tokens}tok_prefill1"
        r = run_test(engine, session, label, short_prompt, tokens)
        all_results[label] = r
        avail, swap = check_memory()
        print(f"    [mem] Avail={avail}MB, SwapUsed={swap}KB")

    # ============ Part 2: Prefill 延迟 - 不同输入长度 ============
    print("\n" + "=" * 100)
    print(" [Part 2/4] Prefill 延迟 vs 输入长度 (max_prefill_length=1)")
    print("=" * 100)

    for name, prompt in [("short", short_prompt), ("medium", medium_prompt), ("long", long_prompt)]:
        label = f"prefill_{name}_pflen1"
        r = run_test(engine, session, label, prompt, 30)
        all_results[label] = r

    # ============ Part 3: Prefill 延迟 - 不同 prefill 分块长度 ============
    print("\n" + "=" * 100)
    print(" [Part 3/4] Prefill 分块长度对 TTFT 的影响 (medium prompt)")
    print("=" * 100)

    del engine
    del session
    import gc; gc.collect()

    for pf_len in [1, 2, 4, 8, 16, 32]:
        engine = create_engine(max_prefill_length=pf_len)
        session = engine.session
        label = f"prefill_medium_pflen{pf_len}"
        r = run_test(engine, session, label, medium_prompt, 30)
        all_results[label] = r
        avail, swap = check_memory()
        print(f"    [mem] Avail={avail}MB, SwapUsed={swap}KB")
        del engine, session
        gc.collect()

    # ============ Part 4: 采样策略对比 ============
    print("\n" + "=" * 100)
    print(" [Part 4/4] 采样策略对比 (CPU greedy vs top_p vs top_k)")
    print("=" * 100)

    engine = create_engine(max_prefill_length=1)
    session = engine.session

    run_once(engine, session, "hello", 5)

    sampling_configs = [
        ("greedy_cpu", "greedy", 0.95, 0.0),
        ("topp08_cpu", "top_p", 0.8, 0.7),
        ("topp095_cpu", "top_p", 0.95, 0.7),
        ("topk50_cpu", "top_k", 50, 0.7),
    ]

    for name, method, value, temp in sampling_configs:
        label = f"sampling_{name}"
        messages = [{"role": "user", "content": medium_prompt}]
        text = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = engine.tokenizer([text], return_tensors="np")["input_ids"].astype(np.int64).reshape(1, -1)
        input_ids = input_ids[:, -engine.max_input_length:]
        input_length = input_ids.shape[1]
        max_out = min(engine.max_output_length - input_length, 100)

        ids_list = []
        current = input_ids
        t_start = time.perf_counter()
        t_first = None
        for i in range(max_out):
            if i == 0:
                session.reset()
            logits = session.run(current)
            if i == 0:
                t_first = time.perf_counter()
            next_tok = engine.sample_logits(logits[0][-1:], method, value, temp)
            next_tok = next_tok.reshape(1, -1)
            if next_tok[0, 0] == engine.tokenizer.eos_token_id:
                break
            ids_list.append(int(next_tok[0, 0]))
            current = next_tok
        t_end = time.perf_counter()

        r = Result(label=label, prompt_tokens=input_length, generated_tokens=len(ids_list))
        r.total_time_ms = (t_end - t_start) * 1000
        if t_first:
            r.ttft_ms = (t_first - t_start) * 1000
            r.prefill_speed = input_length / (r.ttft_ms / 1000) if r.ttft_ms > 0 else 0
        if r.generated_tokens > 1 and t_first:
            dec_t = (t_end - t_first) * 1000
            r.tpot_ms = dec_t / (r.generated_tokens - 1)
            r.decode_speed = (r.generated_tokens - 1) / (dec_t / 1000)
        all_results[label] = r
        print(f"  {label:<38} prompt={r.prompt_tokens:<5} gen={r.generated_tokens:<6} TTFT={r.ttft_ms:<10.1f} TPOT={r.tpot_ms:<8.1f} Decode={r.decode_speed:<6.1f} tok/s  Prefill={r.prefill_speed:<6.1f} tok/s")

    # ============ 汇总 ============
    print("\n" + "=" * 100)
    print(" 汇总表")
    print("=" * 100)
    print(f"  {'测试项':<38} {'Prompt':<7} {'Gen':<6} {'TTFT(ms)':<11} {'TPOT(ms)':<10} {'Decode(t/s)':<12} {'Prefill(t/s)':<13}")
    print("-" * 100)
    for label, r in all_results.items():
        print(f"  {label:<38} {r.prompt_tokens:<7} {r.generated_tokens:<6} {r.ttft_ms:<11.1f} {r.tpot_ms:<10.1f} {r.decode_speed:<12.1f} {r.prefill_speed:<13.1f}")
    print("-" * 100)

    avail, swap = check_memory()
    print(f"\n[最终] MemAvailable={avail}MB, SwapUsed={swap}KB")
    print("测试完成!")


if __name__ == "__main__":
    main()
