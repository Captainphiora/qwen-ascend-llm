"""Profiling 采集脚本：采集 20 个 decode token 的算子级数据"""
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

prompt = "请详细介绍一下机器学习的基本概念和常用算法"
messages = [{"role": "user", "content": prompt}]
text = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = engine.tokenizer([text], return_tensors="np")["input_ids"].astype(np.int64).reshape(1, -1)

MAX_TOKENS = 20
ids_list = []
current = input_ids
for i in range(MAX_TOKENS):
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
