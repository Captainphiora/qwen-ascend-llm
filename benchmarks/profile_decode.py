"""Profiling 采集脚本：采集 decode token 的算子级数据

用法:
    # 直接采集 (需配合 msprof 包裹)
    msprof --output=./profiling_data --application="python benchmarks/profile_decode.py --om_model_path xxx.om"

    # 或手动指定参数
    python benchmarks/profile_decode.py --om_model_path xxx.om --kv_cache_layout BHSD
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['ACL_LOAD_FROM_FILE'] = '1'

from config import InferenceConfig
from utils.inference import Inference

DEFAULT_HF = "/root/models/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_OM = "output/model/DeepSeek-R1-Distill-Qwen-1.5B_4096_1_v4_noexpand_310b.om"

parser = argparse.ArgumentParser()
parser.add_argument("--om_model_path", type=str, default=DEFAULT_OM)
parser.add_argument("--hf_model_dir", type=str, default=DEFAULT_HF)
parser.add_argument("--kv_cache_layout", type=str, default="BSHD", choices=["BSHD", "BHSD"])
parser.add_argument("--max_new_tokens", type=int, default=20)
parser.add_argument("--device_id", type=int, default=0)
args = parser.parse_args()

config = InferenceConfig(
    hf_model_dir=args.hf_model_dir, om_model_path=args.om_model_path, onnx_model_path="",
    session_type="acl", device_id=args.device_id, max_batch=1,
    max_input_length=4095, max_output_length=4096,
    kv_cache_length=4096, max_prefill_length=1,
    dtype="float16", torch_dtype="float16", device_str="npu",
    temperature=0, sampling_method="greedy", sampling_value=0.95, system_prompt="",
    kv_cache_layout=args.kv_cache_layout,
)
engine = Inference(config)
session = engine.session

prompt = "请详细介绍一下机器学习的基本概念和常用算法"
messages = [{"role": "user", "content": prompt}]
text = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = engine.tokenizer([text], return_tensors="np")["input_ids"].astype(np.int64).reshape(1, -1)

ids_list = []
current = input_ids
for i in range(args.max_new_tokens):
    if i == 0:
        session.reset()
    logits = session.run(current)
    tok = engine.sample_logits(logits[0][-1:], "greedy", 0.95, 0).reshape(1, -1)
    if tok[0, 0] == engine.tokenizer.eos_token_id:
        break
    ids_list.append(int(tok[0, 0]))
    current = tok

print(f"Generated {len(ids_list)} tokens")
print("Done - profiling data collected by msprof wrapper")
