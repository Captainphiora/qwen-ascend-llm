import time
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/host-model/cxj/models/Qwen3.6-35B-A3B"
DEVICE = "npu:0" if torch.npu.is_available() else "cpu"


def main():
    print(f"[INFO] device_map = auto (multi-npu)")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    t1 = time.perf_counter()
    print(f"[阶段1] tokenizer 加载: {t1 - t0:.2f}s")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()
    t2 = time.perf_counter()
    print(f"[阶段2] 模型加载: {t2 - t1:.2f}s")
    print(f"[INFO] model class: {model.__class__.__name__}")
    print(f"[INFO] hf_device_map: {getattr(model, 'hf_device_map', None)}")

    messages = [{"role": "user", "content": "用一句话介绍你自己。"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"[DEBUG] prompt:\n{text!r}")
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    t3 = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
        torch.npu.synchronize()
    t4 = time.perf_counter()

    out = generated[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(out, skip_special_tokens=True)
    print(f"[阶段3] 推理(含编译): {t4 - t3:.2f}s")
    print(f"[结果]\n{response}")


if __name__ == "__main__":
    main()
