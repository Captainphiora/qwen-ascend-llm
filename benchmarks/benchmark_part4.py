"""Part 4: 采样策略对比 (单次运行)"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['ACL_LOAD_FROM_FILE'] = '1'

from config import InferenceConfig
from utils.inference import Inference

HF_MODEL_DIR = "/home/chenxinji/models/DeepSeek-R1-Distill-Qwen-1.5B"
OM_MODEL_PATH = "output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om"

config = InferenceConfig(
    hf_model_dir=HF_MODEL_DIR, om_model_path=OM_MODEL_PATH, onnx_model_path="",
    session_type="acl", device_id=0, max_batch=1,
    max_input_length=4095, max_output_length=4096,
    kv_cache_length=4096, max_prefill_length=1,
    dtype="float16", torch_dtype="float16", device_str="npu",
    temperature=0, sampling_method="greedy", sampling_value=0.95, system_prompt="",
)
engine = Inference(config)
session = engine.session

prompt = "请详细介绍一下机器学习的基本概念和常用算法，包括监督学习、无监督学习、强化学习的区别和应用场景"

def check_mem():
    with open('/proc/meminfo') as f:
        lines = f.readlines()
    info = {}
    for line in lines:
        p = line.split()
        info[p[0].rstrip(':')] = int(p[1])
    return info.get('MemAvailable', 0) // 1024, info.get('SwapTotal', 0) - info.get('SwapFree', 0)

def bench_sampling(label, method, value, temp, max_tokens=100):
    messages = [{"role": "user", "content": prompt}]
    text = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = engine.tokenizer([text], return_tensors="np")["input_ids"].astype(np.int64).reshape(1, -1)
    input_ids = input_ids[:, -engine.max_input_length:]
    input_length = input_ids.shape[1]
    max_out = min(engine.max_output_length - input_length, max_tokens)

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

    gen = len(ids_list)
    ttft = (t_first - t_start) * 1000 if t_first else 0
    tpot = 0; decode_speed = 0
    if gen > 1 and t_first:
        dec_t = (t_end - t_first) * 1000
        tpot = dec_t / (gen - 1)
        decode_speed = (gen - 1) / (dec_t / 1000)
    avail, swap_kb = check_mem()
    print(f"  {label:<30} prompt={input_length:<5} gen={gen:<5} TTFT={ttft:<10.1f} TPOT={tpot:<8.1f} Decode={decode_speed:<6.1f} tok/s  Mem={avail}MB Swap={swap_kb}KB")

print("=" * 100)
print(" [Part 4] 采样策略对比 | 310B1 | greedy/top_p/top_k")
print("=" * 100)

# Warmup
session.reset()
messages = [{"role": "user", "content": "hello"}]
text = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = engine.tokenizer([text], return_tensors="np")["input_ids"].astype(np.int64).reshape(1, -1)
for i in range(5):
    if i == 0: session.reset()
    logits = session.run(ids)
    tok = engine.sample_logits(logits[0][-1:], "greedy", 0.95, 0).reshape(1, -1)
    ids = tok
print("[Warmup done]")

bench_sampling("Greedy (CPU argmax)", "greedy", 0.95, 0.0)
bench_sampling("Top-p=0.8 (CPU)", "top_p", 0.8, 0.7)
bench_sampling("Top-p=0.95 (CPU)", "top_p", 0.95, 0.7)
bench_sampling("Top-k=50 (CPU)", "top_k", 50, 0.7)

print("=" * 100)
print("Done!")
