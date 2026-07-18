from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("/mnt/host-model/cxj/models/DeepSeek-R1-Distill-Qwen-1.5B")

with open("../dataset/math500/test.jsonl") as f:
    items = [json.loads(l) for l in f if l.strip()]

lengths = []
for item in items:
    # 模拟 predict() 中的 apply_chat_template
    messages = [{"role": "user", "content": item["problem"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="np")["input_ids"]
    lengths.append((ids.shape[1], item["unique_id"]))

lengths.sort(reverse=True)
print(f"max tokens : {lengths[0][0]}  ({lengths[0][1]})")
print(f"p99 tokens : {lengths[int(len(lengths)*0.01)][0]}")
print(f"mean tokens: {sum(l for l,_ in lengths)//len(lengths)}")