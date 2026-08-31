
import time
import os
import sys

# print("torch")
start_import = time.perf_counter()
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
end_import = time.perf_counter()
print(f"[阶段1] 库导入耗时: {end_import - start_import:.2f} 秒")
device = "npu:0" if torch.npu.is_available() else "cpu"
# device = 'cpu'
print(f"Using device: {device}")

# 2. 记录模型加载时间
# model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

model_path = "/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B"
start_load = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    local_files_only=True, 
    torch_dtype=torch.float16,
    device_map=device,
    attn_implementation="eager"   # flash att 不可用
).eval()
# ).to("npu").eval()
end_load = time.perf_counter()
print(f"[阶段2] 模型加载耗时: {end_load - start_load:.2f} 秒")

# 3. 记录推理/编译耗时
input_text = "给出《出师表》原文"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
input_ids = inputs.input_ids.to(torch.int32).to(device)
# inputs = tokenizer(input_text, return_tensors="pt").to(torch.int32).to(device)

start_inference = time.perf_counter()
# 强制开启同步，确保计时准确覆盖算子编译与执行
with torch.no_grad():
    generated_ids = model.generate(inputs.input_ids, max_new_tokens=1024)
    torch.npu.synchronize() 
end_inference = time.perf_counter()

response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"推理结果: {response}")
print(f"[阶段3] 首次推理(含编译)耗时: {end_inference - start_inference:.2f} 秒")